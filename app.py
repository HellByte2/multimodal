import torch
import ruclip
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = 'ruclip'

clip = ruclip.CLIP.from_pretrained(cache_dir).eval().to(device)
processor = ruclip.RuCLIPProcessor.from_pretrained(cache_dir)

with open("cifar100.txt", "r") as f:
    text = f.read()
    labels_premade = text.split("\n")

def predict(classes, radio, inp):
  labels = classes.split("\n")
  templates = ['{}', 'это {}', 'на картинке {}', 'это {}, домашнее животное']
  imgs = [inp]
  labels = labels if radio == 0 else labels_premade
  dummy_input = processor(text=labels, images=imgs,
                        return_tensors='pt', padding=True)
                        
  image = dummy_input["pixel_values"].to(device)
  text = dummy_input["input_ids"].to(device)
  
  with torch.no_grad():   
    logits_per_image, logits_per_text = clip(text, image)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    probs = probs[0].tolist()
    res = dict(zip(labels, probs))  
  return res
  
gr.Interface(fn=predict, 
             inputs=[
               gr.Textbox(placeholder='Введите классы. Каждый класс на отдельной строчке', label='Классы'), 
               gr.Radio(["Произвольные", "CIFAR-100"], type="index", value='Произвольные', label="Выбор классов"), 
               gr.Image(type="pil", label='Изображение')
             ],
             outputs=gr.Label(num_top_classes=3, label='Результат')).launch(server_name="0.0.0.0", server_port=7000, share=False)          