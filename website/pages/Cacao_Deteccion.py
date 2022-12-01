import streamlit as st
from PIL import Image, ImageDraw
import requests
import io
import ast

url_object = "https://getobject-qejkc246ba-tl.a.run.app"


st.markdown("# Detección de frutos de cacao 🍫")
st.sidebar.markdown("# Cacao 🍫")


image = Image.open('img/cacao.jpg')
print(type(image))
st.image(image, caption='Cacao')


uploaded_file = st.file_uploader("Carga una imagen", key=f'{2}')
bytes_data = None
bounding_image = None
decoded = None

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    decoded = Image.open(io.BytesIO(bytes_data))
    newsize = (256, 256)
    decoded = decoded.resize(newsize)

    buf = io.BytesIO()
    decoded.save(buf, format='JPEG')
    bytes_data = buf.getvalue()
    bounding_image = decoded.copy()

    st.image(decoded, caption='Cacao')
    #print(decoded.shape)
    
if st.button('Detectar cacao'):
    
    if bytes_data is not None:
        with st.spinner('Procesando la imagen...'):
            resp = requests.post(url_object, files={ 'file' : bytes_data })
        pred = resp.json()
        boxes = pred['boxes']
        labels = pred['pred_classes']
        if bounding_image is not None and decoded is not None:
            img1 = ImageDraw.Draw(bounding_image)
            boxes = ast.literal_eval(boxes)
            labels = ast.literal_eval(labels)
            print(boxes, type(boxes))
            print(labels, type(labels))
            for box, label in zip(boxes, labels):
                print(box, type(list(box)), type(box[0]))  
                x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                shape = (x0, y0, x1, y1)
                img1.rectangle(shape, outline ="blue",width=3)
                img1.text((x0+5, y0+5.), text=label,fill='white')
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(decoded, caption='Cacao')
            with col2:
                st.image(bounding_image, caption='Cacao Detectado')
            
            
    else:
        st.write('Por favor sube una image!')