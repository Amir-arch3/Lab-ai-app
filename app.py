
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# تحميل النموذج المدرب
model = tf.keras.models.load_model("model.h5")

st.set_page_config(page_title="تشخيص عينات الدم", layout="centered")
st.title("🔬 تطبيق تشخيص صور خلايا الدم")
st.markdown("قم برفع صورة عينة دم لتحديد حالتها: سليمة أو مصابة.")

uploaded_file = st.file_uploader("📷 اختر صورة العينة", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="📌 صورة العينة", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.success("✅ النتيجة: العينة **سليمة**")
        st.info("لا تظهر مؤشرات غير طبيعية في العينة.")
    else:
        st.error("⚠️ النتيجة: العينة **مصابة**")
        st.warning("تُلاحظ تشوهات قد تدل على وجود عدوى أو خلل. يُنصح بمراجعة مختص.")
