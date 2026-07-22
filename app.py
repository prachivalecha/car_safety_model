
import math
import joblib
import gradio as gr
import os

print("Current Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

MODEL_PATH = os.path.join(os.getcwd(), "car_safety_model.pkl")
print("Looking for model at:", MODEL_PATH)

css = """
/* ===== Premium Dark Theme ===== */

body,
.gradio-container{
    margin:0;
    padding:0;
    font-family:'Segoe UI',sans-serif;
    background:linear-gradient(135deg,#071739,#0F2C59,#123B70);
    color:#FFFFFF;
}

/* Remove default footer */
footer{
    display:none !important;
}

/* Main container */
.gradio-container{
    max-width:1200px !important;
    margin:auto;
    padding:30px !important;
}

/* Glass Cards */

.gr-group,
.gr-box,
.block{
    background:rgba(255,255,255,0.07) !important;
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:22px !important;
    padding:22px !important;
    box-shadow:
        0 8px 25px rgba(0,0,0,.35);
}

/* Headings */

h1{
    font-size:42px !important;
    text-align:center;
    color:#00D4FF;
    margin-bottom:5px;
}

h2,h3{
    color:#5CE1E6;
    text-align:center;
}

/* Markdown */

.markdown{
    color:white;
    line-height:1.8;
}

/* Labels */

label{
    color:#E8F1FF !important;
    font-weight:600 !important;
    font-size:15px !important;
}

/* Inputs */

input,
textarea{
    background:#132F59 !important;
    color:white !important;
    border-radius:12px !important;
    border:1px solid #2B82C9 !important;
    transition:.3s;
}

input:focus,
textarea:focus{
    border-color:#00D4FF !important;
    box-shadow:0 0 15px rgba(0,212,255,.6);
}

/* Buttons */

button{
    background:linear-gradient(90deg,#00D4FF,#5CE1E6)!important;
    color:#071739!important;
    font-size:18px!important;
    font-weight:bold!important;
    border:none!important;
    border-radius:14px!important;
    padding:14px 28px!important;
    transition:all .3s ease!important;
    box-shadow:0 6px 18px rgba(0,212,255,.35);
}

button:hover{
    transform:translateY(-3px);
    box-shadow:0 10px 28px rgba(0,212,255,.55);
}

/* Output */

.prose{
    color:white !important;
    font-size:18px;
    line-height:1.8;
}

/* HR */

hr{
    border:none;
    height:1px;
    background:rgba(255,255,255,.2);
    margin:25px 0;
}

/* Scrollbar */

::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-track{
    background:#071739;
}

::-webkit-scrollbar-thumb{
    background:#00D4FF;
    border-radius:20px;
}

::-webkit-scrollbar-thumb:hover{
    background:#5CE1E6;
}

/* Responsive */

@media(max-width:768px){

    .gradio-container{
        padding:15px !important;
    }

    h1{
        font-size:30px !important;
    }

    button{
        width:100%;
    }

}
"""

model = None
feature_names = []
n_features = 0
model_error = None

try:
    print("=" * 50)
    print("Current Directory:", os.getcwd())
    print("Files:", os.listdir())
    print("Model Path:", MODEL_PATH)
    print("Model Exists:", os.path.exists(MODEL_PATH))
    print("=" * 50)

    model = joblib.load(MODEL_PATH)

    print("✅ Model loaded successfully!")

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if hasattr(model, "n_features_in_"):
        n_features = int(model.n_features_in_)
    elif feature_names:
        n_features = len(feature_names)
    else:
        raise ValueError("Unable to determine feature count.")

    if not feature_names:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

except Exception as e:
    print("MODEL LOADING ERROR:", repr(e))
    model_error = str(e)

def predict(*vals):
    if model_error:
        return f"❌ Model Error\n\n{model_error}"
    try:
        x=[]
        for v in vals:
            if v is None:
                return "⚠️ Please fill all fields."
            v=float(v)
            if math.isnan(v) or math.isinf(v):
                return "⚠️ Invalid numeric value."
            x.append(v)
        pred=model.predict([x])[0]
        out=f"### Prediction\n**Result:** {pred}\n\n"
        if hasattr(model,"predict_proba"):
            try:
                p=model.predict_proba([x])[0]
                conf=max(p)*100
                out+=f"**Confidence:** {conf:.2f}%\n\n"
            except:
                pass
        safe=str(pred).lower() in ["1","safe","yes","true","acceptable","good"]
        if safe:
            out+="✅ **Vehicle appears to be safe based on the provided information.**"
        else:
            out+="⚠️ **Vehicle may not meet safety standards. Please review the input values.**"
        return out
    except Exception:
        return "❌ Unable to generate prediction. Please verify the entered values."

def example():
    return [1.0]*n_features

with gr.Blocks(css=css, title="Car Safety Prediction System") as demo:

    gr.Markdown("""
---
<div style="text-align:center;padding:25px;color:#E8F1FF;">

<h1 style="color:#00D4FF;">🚗 Car Safety Prediction System</h1>

<p>
Predict vehicle safety intelligently using Machine Learning
</p>

<b>Developed by:</b> Prachi Valecha<br>
<b>College:</b> Panipat Institute of Engineering and Technology<br>
<b>Course:</b> Bachelor of Computer Applications (BCA)<br>
<b>Specialization:</b> Cloud Technology & Information Security

</div>
""")

    with gr.Row(equal_height=True):

        with gr.Column(scale=2):

            gr.Markdown("## 🚘 Vehicle Information")

            inputs = []

            for name in feature_names:
                inputs.append(
                    gr.Number(
                        label=f"📌 {name}",
                        placeholder=f"Enter {name}",
                    )
                )

        with gr.Column(scale=1):

            gr.Markdown("## ⚡ Prediction")

            out = gr.Markdown(
                """
### Waiting for Prediction...

Fill in all the details and click **🚀 Predict Safety**.
"""
            )

            btn = gr.Button(
                "🚀 Predict Safety",
                variant="primary"
            )

            ex = gr.Button(
                "💡 Load Example",
                variant="secondary"
            )

            clr = gr.ClearButton(
                inputs + [out],
                value="🧹 Clear"
            )
    btn.click(
        fn=predict,
        inputs=inputs,
        outputs=out,
        show_progress="full"
    )

    ex.click(
        fn=example,
        outputs=inputs
    )

    gr.Markdown("""
---
<div style="text-align:center;padding:20px;">

<h3 style="color:#00D4FF;">
Developed by Prachi Valecha
</h3>

Panipat Institute of Engineering and Technology

Machine Learning Project

© 2026 All Rights Reserved

</div>
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=False
    )
