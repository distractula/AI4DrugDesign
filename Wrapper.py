import gradio as gr
import Core
import os
from dotenv import load_dotenv

# Load and Initialize the GEMINI API
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize The AI AGENT
agent = Core.AIAgent(GEMINI_API_KEY)


# ----------------- Functions ----------------- #

def analyze_protein(pdb_id):
    try:
        result = agent.analyze_target_protein(pdb_id)
        summary = result['summary']
        analysis = result['analysis']
        return summary, analysis
    except Exception as e:
        return f"Error fetching protein: {e}", ""


def evaluate_compound(pdb_id, smiles):
    try:
        result = agent.evaluate_compound(pdb_id, smiles)
        return result
    except Exception as e:
        return f"Error evaluating compound: {e}"


def optimize_compound(smiles, goals):
    try:
        result = agent.optimize_compound(smiles, goals)
        return result
    except Exception as e:
        return f"Error optimizing compound: {e}"
    
def visualize_compound(smiles):
    try:
        img = Core.CompoundAnalyzer.visualize_smiles(smiles)
        return img
    except Exception as e:
        return f"Error visualizing compound: {e}"


# ----------------- Gradio UI ----------------- #

with gr.Blocks() as demo:
    gr.Markdown("## AI Drug Design Platform")

    with gr.Tab("Protein Analysis"):
        with gr.Row():
            with gr.Column(scale=0, min_width=700):
                pdb_input = gr.Textbox(label="Enter PDB ID", placeholder="e.g., 1AZ5")
                submit_protein = gr.Button("Analyze Protein")
            with gr.Column(scale=1):
                protein_summary = gr.Textbox(label="Protein Summary", lines=3, interactive=False)
                protein_analysis = gr.Textbox(label="Analysis", lines=3, interactive=False)

            submit_protein.click(analyze_protein, inputs=pdb_input, outputs=[protein_summary, protein_analysis])

    with gr.Tab("Compound Evaluation"):
        with gr.Row():
            with gr.Column(scale=0, min_width=700):
                pdb_input_eval = gr.Textbox(label="Target Protein PDB ID", placeholder="e.g., 6LU7")
                compound_input = gr.Textbox(label="Compound SMILES", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
                submit_eval = gr.Button("Evaluate Compound")
            with gr.Column(scale=1):
                compound_output = gr.Textbox(label="Evaluation Result", lines=9, interactive=False)

        submit_eval.click(evaluate_compound, inputs=[pdb_input_eval, compound_input], outputs=compound_output)

    with gr.Tab("Compound Optimization"):
        with gr.Row():
            with gr.Column(scale=0, min_width=700):
                compound_input_opt = gr.Textbox(label="Compound SMILES", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
                goals_input = gr.Textbox(label="Optimization Goals", placeholder="Increase solubility, reduce lipophilicity")
                submit_opt = gr.Button("Optimize Compound")
            with gr.Column(scale=1):
                optimization_output = gr.Textbox(label="Optimization Result", lines=9, interactive=False)

        submit_opt.click(optimize_compound, inputs=[compound_input_opt, goals_input], outputs=optimization_output)
        
    with gr.Tab("Compound Visualization"):
        with gr.Row():
            with gr.Column(scale=0, min_width=700):
                smiles_input_vis = gr.Textbox(label="Compound SMILES", placeholder="e.g., C1=CC=CC=C1")
                submit_vis = gr.Button("Visualize Compound")
            with gr.Column():
                img_output = gr.Image(label="Molecule Structure")

        submit_vis.click(visualize_compound, inputs=smiles_input_vis, outputs=img_output)

demo.launch(share=True)