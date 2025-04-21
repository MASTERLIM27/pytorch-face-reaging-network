import torch
import torch.nn.functional as F
from torch.nn.modules import BCEWithLogitsLoss
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
from piq import LPIPS
import torch.nn as nn
import cv2
import os
import gc
from torchvision.models import VGG16_Weights
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import insightface
from insightface.app import FaceAnalysis

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic losses
adversarial_loss = BCEWithLogitsLoss()
l1_loss = nn.L1Loss()
perceptual_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

# Default loss weights
lambda_l1 = 0.5
lambda_perceptual = 0.5
lambda_adversarial = 0.025
lambda_identity = 0.1

# Basic pl lightning module for checkpointing and logging
class FRAN(pl.LightningModule):
  def __init__(self, generator, discriminator):
    super(FRAN, self).__init__()
    self.generator = generator
    self.discriminator =discriminator
    self.automatic_optimization = False

    # Load ArcFace model
    self.face_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
    self.face_model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

  def forward(self, x):
      with torch.no_grad():
        model_output = self.generator(x)
      return model_output
  
  def check_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}!")
        if torch.isinf(tensor).any():
            print(f"Inf detected in {name}!")

  def training_step(self, batch, batch_idx):
      opt_g, opt_d = self.optimizers()

      opt_g.zero_grad()
      opt_d.zero_grad()

      self.clip_gradients(opt_g, gradient_clip_val=1.0)
      self.clip_gradients(opt_d, gradient_clip_val=1.0)

      inputs = batch['input'].to(self.device)
      input_aug_1 = batch['input_aug_1'].to(self.device)
      input_aug_2 = batch['input_aug_2'].to(self.device)

      normalized_input_image = batch['normalized_input_image'].to(self.device)
      normalized_input_aug_1_image = batch['normalized_input_aug_1_image'].to(self.device)
      normalized_input_aug_2_image = batch['normalized_input_aug_2_image'].to(self.device)

      normalized_target_image = batch['normalized_target_image'].to(self.device)
      normalized_target__aug_image = batch['normalized_target__aug_image'].to(self.device)

      target_age = batch['target_age'].to(self.device)

      # Forward pass
      outputs = self.generator(inputs)
      outputs_aug_1 = self.generator(input_aug_1)
      outputs_aug_2 = self.generator(input_aug_2)
      outputs = torch.clamp(outputs, -1, 1)
      outputs_aug_1 = torch.clamp(outputs_aug_1, -1, 1)
      outputs_aug_2 = torch.clamp(outputs_aug_2, -1, 1)

      predicted_images = normalized_input_image + outputs
      predicted_aug_1_images = normalized_input_aug_1_image + outputs_aug_1
      predicted_aug_2_images = normalized_input_aug_2_image + outputs_aug_2
      predicted_images = torch.clamp(predicted_images, -1, 1)
      predicted_aug_1_images = torch.clamp(predicted_aug_1_images, -1, 1)
      predicted_aug_2_images = torch.clamp(predicted_aug_2_images, -1, 1)
      predicted_images_with_age = torch.cat((predicted_images, target_age), dim=1)
      predicted_aug_1_images_with_age = torch.cat((predicted_aug_1_images, target_age), dim=1)
      predicted_aug_2_images_with_age = torch.cat((predicted_aug_2_images, target_age), dim=1)


      # Prepare both images for ArcFace (RGB -> BGR + Resize)
      def preprocess_for_arcface(img_tensor):
          img = ((img_tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
          img = cv2.resize(img, (112, 112))
          return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      img1 = preprocess_for_arcface(predicted_aug_1_images[0])
      img2 = preprocess_for_arcface(predicted_aug_2_images[0])

      def get_embedding_from_image(image_np):
        faces = self.face_model.get(image_np)
        if len(faces) == 0:
            return None
        emb = torch.tensor(faces[0].embedding, dtype=torch.float32)
        return emb
      
      emb1 = get_embedding_from_image(img1)
      emb2 = get_embedding_from_image(img2)

      identity_loss = torch.tensor(0.0, device=self.device)

      # ----------- Identity Loss ----------- #
      if emb1 is not None and emb2 is not None:
          emb1 = F.normalize(emb1.unsqueeze(0), dim=1)
          emb2 = F.normalize(emb2.unsqueeze(0), dim=1)
          identity_loss = 1 - F.cosine_similarity(emb1, emb2).item()
        #   print(f"Identity Loss (1 - cosine similarity): {identity_loss:.4f}")
      else:
          identity_loss = 1
        #   cv2.imwrite(f"debug_no_face_img1_{self.global_step}.png", img1)
        #   cv2.imwrite(f"debug_no_face_img2_{self.global_step}.png", img2)
        #   print("No face detected in one or both images — skipping identity loss this step.")


      # # Prepare both images for ArcFace (RGB -> BGR + Resize)
      # def preprocess_for_arcface(img_tensor):
      #     img = ((img_tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
      #     img = cv2.resize(img, (112, 112))
      #     return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      # img1 = preprocess_for_arcface(predicted_aug_1_images[0])
      # img2 = preprocess_for_arcface(predicted_aug_2_images[0])

      # # Extract ArcFace embeddings
      # faces1 = self.face_model.get(img1)
      # faces2 = self.face_model.get(img2)

      # identity_loss = torch.tensor(0.0, device=self.device)

      # if len(faces1) == 0 or len(faces2) == 0:
      #     print("No face detected in one or both images — skipping identity loss this step.")
      #     identity_loss = torch.tensor(0.0, device=self.device)
      # else:
      #     embed1 = torch.tensor(faces1[0]['embedding'], device=self.device)
      #     embed2 = torch.tensor(faces2[0]['embedding'], device=self.device)
      #     identity_loss = nn.functional.mse_loss(embed1, embed2)

      # if len(faces1) == 0 or len(faces2) == 0:
      #   cv2.imwrite(f"debug_no_face_img1_{self.global_step}.png", img1)
      #   cv2.imwrite(f"debug_no_face_img2_{self.global_step}.png", img2)

      real_labels = torch.full((inputs.shape[0], 1, 32, 32), 0.9, device=self.device)
      fake_labels = torch.full((inputs.shape[0], 1, 32, 32), 0.1, device=self.device)

      # Compute discriminator losses
      real_loss = adversarial_loss(self.discriminator(torch.cat((normalized_target_image, target_age), dim=1)), real_labels)
      fake_loss = adversarial_loss(self.discriminator(predicted_images_with_age.detach()), fake_labels)

      d_loss = (real_loss+fake_loss) / 2

      # Optimize Discriminator
      opt_g, opt_d = self.optimizers()
      opt_g.zero_grad()
      opt_d.zero_grad()
      self.clip_gradients(opt_d, gradient_clip_val=1.0)
      self.manual_backward(d_loss)
      opt_d.step()

      # Compute generator loss
      l1_loss_value = l1_loss(predicted_images, normalized_target_image)
      perceptual_loss_value = perceptual_loss(predicted_images, normalized_target_image)
      adversarial_loss_value = adversarial_loss(self.discriminator(predicted_images_with_age), real_labels)

      total_loss =  lambda_adversarial*adversarial_loss_value+ lambda_perceptual*perceptual_loss_value + lambda_l1*l1_loss_value  + lambda_identity*identity_loss

      # === Optimize Generator ===
      opt_g.zero_grad()
      self.clip_gradients(opt_g, gradient_clip_val=1.0)
      self.manual_backward(total_loss)
      opt_g.step()

      # Log loss
      self.log('fake_loss', fake_loss, prog_bar=True)
      self.log('discriminator_loss', d_loss, prog_bar=True)
      self.log('total_loss', total_loss, prog_bar=True)
      self.log('gen_adversarial_loss', lambda_adversarial*adversarial_loss_value, prog_bar=True)
      self.log('perceptual_loss', perceptual_loss_value.mean()*lambda_perceptual, prog_bar=True)
      self.log('l1_loss', l1_loss_value*lambda_l1, prog_bar=True)
      self.log('identity_loss', identity_loss * lambda_identity, prog_bar=True)

      # Debug checks
      self.check_nan_inf(outputs, "Generator Output")
      self.check_nan_inf(predicted_images, "Predicted Images")
      self.check_nan_inf(d_loss, "Discriminator Loss")
      self.check_nan_inf(total_loss, "Total Loss")

      # Display images every 500 steps     
      if self.global_step % 500 == 0:

        # Define the output directory
        output_dir = os.path.join(self.logger.log_dir, "output_images")
        os.makedirs(output_dir, exist_ok=True)
        sample_image = inputs[0]
        model_output = outputs.detach()
        model_outputs_aug_1 = outputs_aug_1.detach()
        model_outputs_aug_2 = outputs_aug_2.detach()

        # Convert images to NumPy format for saving
        input_img = ((normalized_input_image[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype('uint8')
        target_img = ((normalized_target_image[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype('uint8')
        output_img = (((normalized_input_image[0].cpu() + model_output[0].cpu()).permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        tar_diff_img = (((torch.abs(normalized_target_image[0].cpu() - normalized_input_image[0].cpu())).permute(1, 2, 0).numpy() + 1) * 127.5).astype('uint8')
        pred_diff_img = (((torch.abs(model_output[0].cpu())).permute(1, 2, 0).numpy() + 1) * 127.5).astype('uint8')
        input_aug_1_img = ((normalized_input_aug_1_image[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        input_aug_2_img = ((normalized_input_aug_2_image[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        output_aug_1_input_img = (((normalized_input_aug_1_image[0].cpu() + model_outputs_aug_1[0].cpu()).permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        output_aug_2_input_img = (((normalized_input_aug_2_image[0].cpu() + model_outputs_aug_2[0].cpu()).permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')


        print(f"D_real min: {real_loss.min().item()}, max: {real_loss.max().item()}")
        print(f"D_fake min: {fake_loss.min().item()}, max: {fake_loss.max().item()}")

        print(f"Input min: {inputs.min().item()}, max: {inputs.max().item()}")
        print(f"Output min: {outputs.min().item()}, max: {outputs.max().item()}")
        print(f"Normalized Input min: {normalized_input_image.min().item()}, max: {normalized_input_image.max().item()}")
        print(f"Normalized Target min: {normalized_target_image.min().item()}, max: {normalized_target_image.max().item()}")
        print(f"Predicted Image min: {predicted_images.min().item()}, max: {predicted_images.max().item()}")

        print(f"Step {self.global_step}: D Loss: {d_loss.item()}, G Loss: {total_loss.item()}")
        print(f"L1 Loss: {l1_loss_value.item()}, Perceptual Loss: {perceptual_loss_value.item()}, Adv Loss: {adversarial_loss_value.item()}, Identity Loss: {identity_loss.item()}")

        if torch.isnan(outputs).any():
            print("NaN detected in Generator output!")
        if torch.isnan(predicted_images).any():
            print("NaN detected in predicted images!")
        if torch.isnan(d_loss).any() or torch.isnan(total_loss).any():
            print("NaN detected in loss values!")
          
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 9, 1)
        plt.imshow(input_img)
        plt.title(f"Input, Age: {int(sample_image[3][0][0]*100)}")
        plt.axis('off')

        plt.subplot(1, 9, 2)
        plt.imshow(target_img)
        plt.title("Target")
        plt.axis('off')

        print(f"Target Image min: {target_img.min().item()}, max: {target_img.max().item()}")

        plt.subplot(1, 9, 3)
        plt.imshow(output_img)
        plt.title(f"Output, Age: {int(sample_image[4][0][0]*100)}")
        plt.axis('off')

        print(f"Output Image min: {output_img.min().item()}, max: {output_img.max().item()}")

        plt.subplot(1, 9, 4)
        plt.imshow(tar_diff_img)
        plt.title(f"Tar RGB Diff")
        plt.axis('off')

        plt.subplot(1, 9, 5)
        plt.imshow(pred_diff_img)
        plt.title(f"Pred RGB Diff")
        plt.axis('off')

        plt.subplot(1, 9, 6)
        plt.imshow(input_aug_1_img)
        plt.title("Aug 1 Input")
        plt.axis('off')

        plt.subplot(1, 9, 7)
        plt.imshow(output_aug_1_input_img)
        plt.title("Aug 1 Output")
        plt.axis('off')

        plt.subplot(1, 9, 8)
        plt.imshow(input_aug_2_img)
        plt.title("Aug 2 Input")
        plt.axis('off')

        plt.subplot(1, 9, 9)
        plt.imshow(output_aug_2_input_img)
        plt.title("Aug 2 Output")
        plt.axis('off')

        # Define the output directory
        output_step_dir = os.path.join(output_dir, f"step_{self.global_step}")
        os.makedirs(output_step_dir, exist_ok=True)

        # Save individual images
        plt.imsave(os.path.join(output_step_dir, "input.png"), input_img)
        plt.imsave(os.path.join(output_step_dir, "target.png"), target_img)
        plt.imsave(os.path.join(output_step_dir, "output.png"), output_img)
        plt.imsave(os.path.join(output_step_dir, "target_diff.png"), tar_diff_img)
        plt.imsave(os.path.join(output_step_dir, "pred_diff.png"), pred_diff_img)
        plt.imsave(os.path.join(output_step_dir, "augmented_1_input.png"), input_aug_1_img)
        plt.imsave(os.path.join(output_step_dir, "augmented_2_input.png"), input_aug_2_img)
        plt.imsave(os.path.join(output_step_dir, "augmented_1_output.png"), output_aug_1_input_img)
        plt.imsave(os.path.join(output_step_dir, "augmented_2_output.png"), output_aug_2_input_img)

        # Save the full figure
        plt.savefig(os.path.join(output_step_dir, f"comparison_{self.global_step}.png"), bbox_inches='tight')
        plt.close()

  def configure_optimizers(self):
    generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
    discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)

    return [generator_optimizer, discriminator_optimizer]