import torch
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

def load_model(model, model_weights):
  weights = model_weights.DEFAULT
  model = model(weights=weights).eval()
  preprocess = weights.transforms()
  return model, weights, preprocess

def load_image(path: str, preprocess):
  image = decode_image(path)
  input_tensor = preprocess(image).unsqueeze(0)
  return image, input_tensor

def generate_cam(extractor, model, tensor, weights, target_layer: str = None):
  with extractor(model, target_layer=target_layer) as cam_extractor:
    out = model(tensor)
    class_idx = out.squeeze(0).argmax().item()
    class_name = weights.meta["categories"][class_idx]
    activation_map = cam_extractor(class_idx, out)
  return class_idx, class_name, activation_map

def cam_overlay_on_image(image, activation_map, alpha: float = 0.65):
  cam_overlayed_image = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode="F"), alpha=alpha)
  return cam_overlayed_image

def plot_and_predict(original_image, cams, class_name, class_idx):
  _, axes = plt.subplots(1, len(cams)+1, figsize=((len(cams)+1)*4, 4))

  axes[0].imshow(original_image.permute(1, 2, 0))
  axes[0].set_title("Original image")
  axes[0].axis("off")

  for i, layer in enumerate(cams, start=1):
      axes[i].imshow(cams[layer])
      axes[i].set_title(f"CAM {layer}")
      axes[i].axis("off")

  plt.suptitle(f"PREDICTED CLASS: {class_name.upper()} (index {class_idx})", fontsize=16)
  plt.tight_layout()
  plt.show()

def extract_logits_and_confidences(model, tensor, weights):
  #> copilot.microsoft.com
  #> Skapa en funktion som tar emot en modell, en tensor och förtränade vikter och 
  #  sedan skriver ut topp 5 klasser med tillhörande logits och softmax-confidence.
  
  with torch.no_grad():
    logits = model(tensor)[0]
    probabilities = torch.softmax(logits, dim=0)

  top5 = torch.topk(probabilities, 5)

  print(f"{'CLASS':18s} {'LOGIT':>10s} {'CONFIDENCE':>12s}")
  for idx, prob in zip(top5.indices, top5.values):
      class_name = weights.meta["categories"][idx]
      logit = logits[idx].item()
      print(f"{class_name:18s} {logit:10.4f} {prob.item():12.5f}")

def pipeline(model, weights, preprocess, extractor, image_path, layers=["layer4"]):
  original, tensor = load_image(image_path, preprocess)
  cams = {}
  for layer in layers:
      class_idx, class_name, cam = generate_cam(extractor, model, tensor, weights, layer)
      cams[layer] = cam_overlay_on_image(original, cam)
  plot_and_predict(original, cams, class_name, class_idx)
  extract_logits_and_confidences(model, tensor, weights)

def extract_class(path):
  return (path.split("/")[-1]
          .split(".")[0]
          .replace("_", " ")
          .upper())

def plot_for_comparison(path, preprocess):
  original_image, _ = load_image(path, preprocess)
  plt.figure(figsize=(4, 8))
  plt.imshow(original_image.permute(1, 2, 0))
  plt.title(f"{extract_class(path)}")
  plt.axis("off")
  plt.tight_layout()
  plt.show()