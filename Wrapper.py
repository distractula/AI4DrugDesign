import gradio as gr
import Core
import os
from dotenv import load_dotenv
import json

# Load and Initialize the GEMINI API
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize The AI AGENT
# agent = Core.AIAgent(GEMINI_API_KEY)
agent = Core.AIAgent()


# ----------------- Functions ----------------- #

def analyze_protein(pdb_id):
    try:
        # Get static summary and streaming analysis generator from agent
        result = agent.analyze_target_protein(pdb_id)
        summary_text = result["summary"]       # static string
        analysis_gen = result["analysis"]      # generator

        analysis_accum = ""

        # FIRST YIELD: static summary, empty analysis
        yield summary_text, ""

        # STREAMING YIELDS: summary stays fixed, analysis grows
        for chunk in analysis_gen:
            analysis_accum += chunk
            yield summary_text, analysis_accum

    except Exception as e:
        # In error case, just yield once
        yield f"Error fetching protein: {e}", ""


def evaluate_compound(pdb_id, smiles):
    try:
        result = agent.evaluate_compound(pdb_id, smiles)
        out = ""

        for chunk in result:
            out += chunk
            yield out

    except Exception as e:
        return f"Error evaluating compound: {e}"


def optimize_compound(smiles, goals):
    try:
        result = agent.optimize_compound(smiles, goals)
        out = ""

        for chunk in result:
            out += chunk
            yield out

    except Exception as e:
        return f"Error optimizing compound: {e}"
    
def visualize_compound(smiles):
    try:
        img = Core.CompoundAnalyzer.visualize_smiles(smiles)
        return img
    except Exception as e:
        return f"Error visualizing compound: {e}"
    
    
def autofill_coordinates(pdb_id: str) -> tuple[float, float, float, str]:
    """
    Downloads the PDB, calculates the center of the largest HETATM, and returns the coordinates.
    """
    if not pdb_id or len(pdb_id.strip()) == 0:
        return 0.0, 0.0, 0.0, "Enter a PDB ID to auto-calculate the binding center."

    try:
        # 1. Download the raw PDB (necessary to find HETATMs)
        # We reuse the logic from get_protein_pdb_path but need the path
        pdb_raw_path = Core.DockingTools.get_protein_pdb_path(pdb_id, output_pdb_path=f"{pdb_id}_raw_center.pdb")
        
        # 2. Calculate the center
        center_x, center_y, center_z = Core.DockingTools.calculate_ligand_center(pdb_id, pdb_raw_path)

        # 3. Clean up the temporary file (optional but good practice)
        os.remove(pdb_raw_path)

        if center_x == 0.0 and center_y == 0.0 and center_z == 0.0:
            msg = "No non-water ligand found. Defaulting to (0, 0, 0). **Manual entry is required!**"
        else:
            msg = f"Center calculated from co-crystallized ligand in {pdb_id}."
            
        return center_x, center_y, center_z, msg
        
    except Exception as e:
        error_msg = f"Coordinate calculation failed: {e}. Please enter coordinates manually."
        return 0.0, 0.0, 0.0, error_msg


def prepare_for_docking(pdb_id: str, smiles: str) -> str:
    """
    Coordinates the retrievals and preparation of protein and ligand files.
    """
    try:
        protein_path = Core.DockingTools.get_protein_pdb_path(pdb_id, output_pdb_path=f"{pdb_id}_protein.pdb")
        
        ligand_path = Core.DockingTools.prepare_ligand_3d(smiles, output_pdb_path="user_ligand.pdb")
        return f"""
        ### ✅ Preparation Successful!

        * **Protein PDB saved:** `{protein_path}`
        * **Ligand PDB (3D Conformer) saved:** `{ligand_path}`
        
        To proceed with **AutoDock Vina**, you must now perform these external steps:
        
        1.  **PDBQT Conversion:** Convert the `{protein_path}` and `{ligand_path}` files to PDBQT format using tools like **MGLTools** (`prepare_receptor4.py`, `prepare_ligand4.py`) or **OpenBabel**.
        2.  **Binding Box Definition:** Determine the binding site center coordinates and box size (typically around 20-30 Å) in the PDBQT file's coordinate system.
        3.  **Run Vina:** Execute the Vina command line tool using the generated PDBQT files and binding box parameters.
        """
    except Exception as e:
        return f"Error during docking preparation: {e}"
    

def full_docking_workflow(pdb_id: str, smiles: str, center_x: float, center_y: float, center_z: float) -> tuple[str, str, str]:
    """
    Executes the full Vina pipeline: Prep -> PDBQT Conversion -> Config -> Docking -> Visualization.
    
    Returns: HTML for 3D Viewer, Docking Results Text, Docked PDBQT File Path (for display in interface)
    """
    output_base_name = f"{pdb_id}_{os.path.basename(smiles).split('.')[0]}"
    
    try:
        protein_raw_path = Core.DockingTools.get_protein_pdb_path(pdb_id, output_pdb_path=f"{pdb_id}_raw.pdb")
        
        protein_pdb_clean = Core.DockingTools.clean_protein_pdb(protein_raw_path, f"{pdb_id}_protein_clean.pdb")
        ligand_pdb = Core.DockingTools.prepare_ligand_3d(smiles, output_pdb_path="user_ligand.pdb")

        protein_pdbqt = Core.DockingTools.convert_to_pdbqt(protein_pdb_clean, output_path=f"{output_base_name}_receptor.pdbqt", is_receptor=True)
        
        ligand_pdbqt = Core.DockingTools.convert_to_pdbqt(ligand_pdb, output_path=f"{output_base_name}_ligand.pdbqt", is_receptor=False)

        config_file = Core.DockingTools.generate_vina_config(
            center_x=float(center_x), 
            center_y=float(center_y), 
            center_z=float(center_z), 
            size=22.0, 
            filename=f"{output_base_name}_config.txt"
        )

        docked_pdbqt_path, affinity_data = Core.DockingTools.run_vina_docking(
            receptor_pdbqt_path=protein_pdbqt,
            ligand_pdbqt_path=ligand_pdbqt,
            config_file_path=config_file,
            output_base=output_base_name
        )

        # with open(docked_pdbqt_path, 'r') as f:
            # docked_pdbqt_content = f.read()

        # html_viewer = visualize_docked_complex_3d(docked_pdbqt_content)
        
        html_viewer = f"""
        <h4>3D Viewer Inoperable </h4>
        <p>The internal 3D viewer failed to load the WebGL canvas. The docking run was successful!</p>
        <p>Please download the result and view it in external software (e.g., PyMOL, ChimeraX):</p>
        <p>Docked file path: <code>{docked_pdbqt_path}</code></p>
        """
        
        return html_viewer, affinity_data, docked_pdbqt_path

    except Exception as e:
        error_msg = f"Docking Error: {e}"
        return f"<h1>Error</h1><p>{error_msg}</p>", error_msg, ""

def get_docked_file_for_download(docked_file_path: str):
    """
    Returns the file path for a Gradio gr.File component.
    """
    if os.path.exists(docked_file_path):
        return gr.File(value=docked_file_path, label="Download Docked PDBQT File")
    return None
# ----------------- Gradio UI ----------------- #

with gr.Blocks() as demo:
    gr.Markdown("## AI Drug Design Platform")

    with gr.Tab("Protein Analysis"):
        with gr.Row(equal_height=True):
            with gr.Column():
                pdb_input = gr.Textbox(label="Enter PDB ID", placeholder="e.g., 1AZ5")
                submit_protein = gr.Button("Analyze Protein")
            with gr.Column():
                protein_summary = gr.Textbox(label="Protein Summary", lines=11, interactive=False)
        with gr.Row():
            protein_analysis = gr.Markdown(label="Analysis")

        submit_protein.click(analyze_protein, inputs=pdb_input, outputs=[protein_summary, protein_analysis])

    with gr.Tab("Compound Evaluation"):
        with gr.Row(equal_height=True):
            with gr.Column():
                pdb_input_eval = gr.Textbox(label="Target Protein PDB ID", placeholder="e.g., 6LU7")
                compound_input = gr.Textbox(label="Compound SMILES", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
            with gr.Column():
                submit_eval = gr.Button("Evaluate Compound")
        with gr.Row():
            compound_output = gr.Markdown(label="Evaluation Result")

        submit_eval.click(evaluate_compound, inputs=[pdb_input_eval, compound_input], outputs=compound_output)

    with gr.Tab("Compound Optimization"):
        with gr.Row(equal_height=True):
            with gr.Column():
                compound_input_opt = gr.Textbox(label="Compound SMILES", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
                goals_input = gr.Textbox(label="Optimization Goals", placeholder="Increase solubility, reduce lipophilicity")
            with gr.Column():
                submit_opt = gr.Button("Optimize Compound")
        with gr.Row():
            optimization_output = gr.Markdown(label="Optimization Result")

        submit_opt.click(optimize_compound, inputs=[compound_input_opt, goals_input], outputs=optimization_output)
        
    with gr.Tab("Compound Visualization"):
        with gr.Row(equal_height=True):
            with gr.Column():
                smiles_input_vis = gr.Textbox(label="Compound SMILES", placeholder="e.g., C1=CC=CC=C1")
                submit_vis = gr.Button("Visualize Compound")
            with gr.Column():
                img_output = gr.Image(label="Molecule Structure")

        submit_vis.click(visualize_compound, inputs=smiles_input_vis, outputs=img_output)
     
    with gr.Tab("Docking Preparation"):
        gr.Markdown('### Autodock Vina Preparation')
        with gr.Row(equal_height=True):
            with gr.Column():
                pdb_dock_input = gr.Textbox(label="Target Protein PDB ID", placeholder="e.g., 6LU7")
                smiles_dock_input = gr.Textbox(label="Compound SMILES", placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O")
            with gr.Column():
                submit_prep = gr.Button("Prepare Files for Vina")
        with gr.Row():
            docking_output = gr.Markdown("Preparation status will appear here.")
        submit_prep.click(prepare_for_docking, inputs=[pdb_dock_input, smiles_dock_input], outputs=docking_output)
        
    with gr.Tab("Molecular Docking"): 
        gr.Markdown('### AutoDock Vina Docking')
        with gr.Row():
            pdb_dock_input = gr.Textbox(label="Target PDB ID", placeholder="e.g., 6LU7")
            smiles_dock_input = gr.Textbox(label="Ligand SMILES", placeholder="e.g., C(=O)Oc1ccccc1C(=O)O")
        gr.Markdown("#### 📦 Binding Box Coordinates (Required)")
        
        autofill_btn = gr.Button("Auto-Calculate Center from Co-Ligand")
        autofill_status = gr.Markdown("Status: Ready for input.")
        
        with gr.Row():
            center_x = gr.Number(label="Center X", value=0.0, interactive=True)
            center_y = gr.Number(label="Center Y", value=0.0, interactive=True)
            center_z = gr.Number(label="Center Z", value=0.0, interactive=True)    
        submit_dock = gr.Button("Run AutoDock Vina")
        docking_result_text = gr.Textbox(label="Vina Binding Affinities (stdout)", lines=10, interactive=False)
        # docked_file_path = gr.Textbox(label="Docked File Output Path", lines=1, interactive=False)
        docked_file_path = gr.Textbox(visible=False)
        docked_file_download = gr.File(label="Download Docked PDBQT File (View in PyMOL/ChimeraX)", interactive=False)
        viewer_output = gr.HTML(label="3D Docked Complex", value="Run docking to generate visualization.")
        autofill_btn.click(autofill_coordinates, inputs=[pdb_dock_input], outputs=[center_x, center_y, center_z, autofill_status])
        submit_dock.click(full_docking_workflow, inputs=[pdb_dock_input, smiles_dock_input, center_x, center_y, center_z], outputs=[viewer_output, docking_result_text, docked_file_path]).success(
            get_docked_file_for_download, inputs=[docked_file_path], outputs=[docked_file_download]
        )

demo.launch(share=True)