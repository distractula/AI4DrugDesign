# import google.generativeai as genai
from google import genai
# import requests
import io
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw, AllChem
from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToMolBlock
import re
import os
import subprocess
import numpy as np

def strip_md(md_text: str) -> str:
    if not isinstance(md_text, str):
        return ""
    # remove bold/italic/code markers
    return re.sub(r"[*_`#>-]", "", md_text)


class PDBAnalyzer:
    PDB_BASE = "https://files.rcsb.org/download/"

    @staticmethod
    def summarize_pdb(pdb_id: str) -> str:
        """
        Retrieves a PDB file from RCSB and returns a human-readable summary along
        with binding/catalytic residues, biological assembly, and missing residues.

        Returns:
            dict: {
                'Protein Name': str,
                'Title': str,
                'Classification': str,
                'Organism': str,
                'Chains': list[str],
                'Keywords': str,
                'Biological Assembly': str,
                'Binding Sites': list[tuple(chain, residue_number, residue_name)],
                'Missing Residues': list[tuple(chain, residue_number, residue_name)]
            }
        """
        # url = f"{PDBAnalyzer.PDB_BASE}{pdb_id.strip()}.pdb"
        pdb_id = pdb_id.strip().upper()
        pdb_data = None
        local_file_path = None
        
        try:
            pdbl = PDBList()
            local_file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format='pdb', overwrite=True)
            
            if not local_file_path or not os.path.exists(local_file_path):
                return f"Error: Bio.PDB failed to retrieve structure {pdb_id}. Check ID or network."
            
            with open(local_file_path, "r") as f:
                pdb_data = f.read()
                
        except Exception as e:
            return f"Error: Critical failure during PDB retrieval for {pdb_id}. Details {e}"

        
        summary = {
            "Protein Name": None,
            "Title": None,
            "Classification": None,
            "Organism": None,
            "Chains": [],
            "Keywords": None,
            "Biological Assembly": None,
            "Binding Sites": [],
            "Missing Residues": [],
        }

        known_residues = [
            "ASP",
            "HIS",
            "SER",
            "GLU",
            "LYS",
            "CYS",
            "THR",
            "TYR",
            "PHE",
            "ILE",
            "LEU",
            "VAL",
            "GLN",
            "ASN",
            "MET",
            "ALA",
            "PRO",
            "TRP",
            "ARG",
            "GLY",
        ]

        # --- Step 1: Parse header lines ---
        for line in pdb_data.splitlines():
            if line.startswith("HEADER"):
                summary["Classification"] = line[10:50].strip()
            elif line.startswith("TITLE"):
                if summary["Title"] is None:
                    summary["Title"] = line[10:].strip()
                else:
                    summary["Title"] += " " + line[10:].strip()
            elif line.startswith("COMPND"):
                if "MOLECULE:" in line:
                    mol_name = line.split("MOLECULE:")[1].split(";")[0].strip()
                    summary["Protein Name"] = mol_name
            elif line.startswith("SOURCE"):
                if "ORGANISM_SCIENTIFIC:" in line:
                    org = line.split("ORGANISM_SCIENTIFIC:")[1].split(";")[0].strip()
                    summary["Organism"] = org
            elif line.startswith("KEYWDS"):
                summary["Keywords"] = line[10:].strip()
            elif line.startswith("REMARK 350"):
                if (
                    "BIOLOGICALLY SIGNIFICANT OLIGOMERIZATION" in line
                    or "AUTHOR DETERMINED BIOLOGICAL UNIT:" in line
                ):
                    if "DIMER" in line.upper():
                        summary["Biological Assembly"] = "Dimer"
                    elif "MONOMER" in line.upper():
                        summary["Biological Assembly"] = "Monomer"
                    else:
                        summary["Biological Assembly"] = line[10:].strip()
            elif line.startswith("SITE") or line.startswith("REMARK 800"):
                for res_name in known_residues:
                    if res_name in line:
                        parts = line.split()
                        try:
                            idx = parts.index(res_name)
                            res = res_name
                            chain = parts[idx + 1]
                            res_num = int(parts[idx + 2])
                            summary["Binding Sites"].append((chain, res_num, res))
                        except (ValueError, IndexError):
                            continue
            elif line.startswith("REMARK 465"):
                for res_name in known_residues:
                    if res_name in line:
                        parts = line.split()
                        try:
                            idx = parts.index(res_name)
                            res = res_name
                            chain = parts[idx + 1]
                            res_num = int(parts[idx + 2])
                            summary["Missing Residues"].append((chain, res_num, res))
                        except (ValueError, IndexError):
                            continue

        # --- Step 2: Parse structure to get chains ---
        chains = []
        try:
            if not isinstance(pdb_data, str) or len(pdb_data.strip()) < 100 or 'END' not in pdb_data:
                raise ValueError("PDB data for {pdb_id} is incomplete or invalid after all attempts.")
            
            
            pdb_io = io.StringIO(pdb_data)
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, pdb_io)
            chains = [chain.id for chain in structure.get_chains()]
            summary["Chains"] = chains
            
        except PDBConstructionException as e:
            return f"Error: PDB structure {pdb_id} is corrupt or non-standard and could not be parsed."
        except Exception as e:
            return f"Error: Unexpected parsing error occured for {pdb_id}. Details: {e}"
        
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

        out = ""
        for key, value in summary.items():
            out += f"{key}: {value}\n"

        return out


class CompoundAnalyzer:
    @staticmethod
    def calculate_basic_properties(smiles: str) -> dict:
        """
        Calculate core molecular descriptors for a given SMILES string.
        """
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            raise ValueError("Input for SMILES is not a valid string.")
        
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        props = {}
        props["MolecularWeight"] = Descriptors.ExactMolWt(mol)
        props["LogP"] = Crippen.MolLogP(mol)
        props["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
        props["HBD"] = rdMolDescriptors.CalcNumHBD(mol)
        props["HBA"] = rdMolDescriptors.CalcNumHBA(mol)
        props["RotatableBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        props["AromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        props["HeavyAtoms"] = Descriptors.HeavyAtomCount(mol)
        props["FractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
        props["ChiralCenters"] = len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        )
        return props

    @staticmethod
    def predict_adme(smiles: str) -> dict:
        """
        Predict basic ADME properties using simple rule-based heuristics.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        adme = {}
        adme["OralBioavailability"] = (
            "High"
            if Descriptors.MolWt(mol) < 500 and Crippen.MolLogP(mol) < 5
            else "Low"
        )
        adme["CYP_Inhibition"] = (
            "Possible" if Descriptors.HeavyAtomCount(mol) > 25 else "Unlikely"
        )
        adme["hERG_Inhibition"] = "Possible" if Crippen.MolLogP(mol) > 4 else "Unlikely"
        adme["SolubilityClass"] = (
            "Soluble" if rdMolDescriptors.CalcTPSA(mol) > 75 else "Poorly Soluble"
        )
        adme["PGP_Substrate"] = "Likely" if Crippen.MolLogP(mol) > 3 else "Unlikely"
        adme["Permeability"] = "High" if Descriptors.MolWt(mol) < 400 else "Low"
        adme["Druglikeness_Score"] = Descriptors.qed(mol)
        return adme

    @staticmethod
    def calculate_all_properties(smiles: str) -> str:
        """
        Calculate all molecular and ADME properties for a given SMILES string.
        """
        result = CompoundAnalyzer.calculate_basic_properties(smiles)
        adme = CompoundAnalyzer.predict_adme(smiles)
        result.update(adme)

        out = ""
        for key, value in result.items():
            out += f"{key}: {value}\n"
        return out
    '''
    @staticmethod
    def generate_ligand_pdb_string(smiles: str) -> str:
        """
        Generates a single 3D conformer for a SMILES string and returns it as a PDB block.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "ERROR: Could not parse SMILES."
            
            mol = Chem.AddHs(mol)
            
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            for atom in mol.GetAtoms():
                atom.SetProp('PDBResName', 'LIG')
                atom.SetProp('PDBResSeq', '900')
                
            pdb_block = Chem.MolToDBBlock(mol)
            return pdb_block.replace("COMPND", "HETATM")
        except Exception as e:
            return f"Error during 3D generation: {e}"
        '''

    @staticmethod
    def visualize_smiles(smiles: str, size=(300, 300)):
        """
        Generate and return an RDkit image of the molecule from SMILES.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        return Draw.MolToImage(mol, size=size)


class AIAgent:
    def __init__(self):
        """
        Initialize the agent and connect to OpenAI's API

        """
        self.TEMPERATURE = 0.7
        self.MAX_TOKENS = 10000
        self.MODEL = "gemini-2.5-flash"
        self.PDB_BASE = "https://files.rcsb.org/download/"
        self.SYSTEM = {
            "role": "system",
            "content": "You are an experienced drug-designer who specializes in structure based drud-design.",
        }

        self.client = genai.Client()

    def reopen_client(self):
        self.client = genai.Client()

    def stream_generate(self, prompt):
        response = self.client.models.generate_content_stream(
            model=self.MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                system_instruction=self.SYSTEM["content"],
                max_output_tokens=self.MAX_TOKENS,
            )
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def analyze_target_protein(self, pdb_id):
        # Provide a high level summary of desired protein
        protein_summary = PDBAnalyzer.summarize_pdb(pdb_id)

        # Check if client is None and reopen if needed
        if self.client is None:
            self.reopen_client()

        prompt = f"""Analyze the following protein and provide insights into potential drug discovery surrounding it:
                    {protein_summary}
                    Provide a detailed analysis that includes the following:
                    1. Binding Sites: Identify main ligand-binding cavities, their size, shape, and accessibility.
                    2. Key Residues: List amino acids crucial for ligand interaction, including hydrogen bond donors/acceptors, charged or hydrophobic residues.
                    3. Interaction Types: Descripe possible chemical interactions (hydrogen bonds, ionic, pi-pi stacking, van der Waals, metal coordination).
                    4. Design Considerations: Highlight flexible regions, allosteric sites, post-translational modifications, or mutation-sensitive residues affecting drug binding.
                    5. Ligand Features: Recommend chemical properties likely to improve binding: functional groups, polarity, hydrophobicity, scaffold shape, size, and rigidity.
                    6. Drug Development Insights: Suggest potential resistance mechanisms, selectivity strategies, off-target risks, and opportunities for covalent/non-covalent modulation.
                    7. Analysis Recommendations: Indicate useful computational (docking, MD, free energy) and experimental (mutagenesis, binding assays, NMR/cryo) approaches to validate and optimize ligands."""

        result = {
            "summary": protein_summary,
            "analysis": self.stream_generate(prompt),
        }

        return result

    def optimize_compound(self, smiles, goals):
        
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            return "Error: Invalid or empty SMILES string provided for compound optimization."
        # Calculate molecule's initial properties
        initial_properties = CompoundAnalyzer.calculate_all_properties(smiles)

        # Check if client is None and reopen if necessary
        if self.client is None:
            self.reopen_client()

        prompt = f"""Optimize the following chemical compound for the following goals:
                    Compound Structure (SMILES): {smiles}
                    Goals: {goals}
                    Initial Properties:
                    {initial_properties}
                    Provide:
                    1. Suggested Structural Modifications
                        -Provide 3-5 specific chemical modifications to the parent compound.
                        For each modification, include:
                            -Chemical Rationale: Explain why this change is proposed (e.g., increase solubility, reduce lipophilicity, improve binding affinity, reduce metabolism).
                            -Expected Property Changes: Predict how key molecular properties will be affected, such as:
                                - Molecular weight
                                - LogP / lipophilicity
                                - TPSA
                                - Hydrogen bond donors/acceptors
                                - Rotatable Bonds
                                - Drug-likeness score (QED or similar)
                                - ADME properties (oral bioavailability, permeability, CYP inhibition, etc.)
                            - Modified SMILES Structure: Provide the exact SMILES string reflecting the proposed modification.
                    2. Optimization Goal Mapping
                        - For each structural modification, explain how it addresses the original optimization goals, e.g.:
                            - Enhanced target binding or specificity
                            - Improved pharmacokinetic profile
                            - Reduced toxicity or off-target interactions
                            - Increased chemical stability or metabolic resistance
                    3. Drug-Likeness Consideration
                        -Prioritize modifications that maintain or improve overall drug-likeness, including adherence to Lipinski's Rule of Five, QED score, and general ADME properties.
                        -Highlight any trade-offs where a modification may improve one property but slightly reduce another, and provide a rationale for why the trade-off is acceptable."""

        return self.stream_generate(prompt)

    def evaluate_compound(self, pdb_id, compound):
        protein_summary = PDBAnalyzer.summarize_pdb(pdb_id)
        if protein_summary.startswith("Error"):
            return protein_summary
        compound_properties = CompoundAnalyzer.calculate_all_properties(compound)

        # Check if client is None and reopen if necessary
        if self.client is None:
            self.reopen_client()

        prompt = f"""Evaluate the following compound for the following target protein.
                    Target:
                    {protein_summary}
                    
                    Compound:
                    SMILES: {compound}
                    {compound_properties}

                    Evaluate the following:
                    1. Likelihood of target binding
                    2. Potential activity against the target
                    3. Pharmacokinetic considerations
                    4. Structural improvements for better target activity
                    5. Potential off-target concerns
                    6. Overall suitability for further development"""

        return self.stream_generate(prompt)
    
class DockingTools:
        
    @staticmethod
    def prepare_ligand_3d(smiles: str, output_pdb_path: str = "ligand.pdb") -> str:
        """
        Generates a 3D conformer for a SMILES string, minimizes it, and saves it
        as a PDB file, ready for PDBQT conversion (e.g., using openbabel/MGLTools).
        
        Returns the path to the saved PDB file.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Could not parse SMILES.")
                
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                
            AllChem.MMFFOptimizeMolecule(mol)
                
            writer = Chem.PDBWriter(output_pdb_path)
            writer.write(mol)
            writer.close()
                
            return output_pdb_path
            
        except Exception as e:
            raise RuntimeError(f"Error during ligand 3D preparation: {e}")
        
    @staticmethod
    def get_protein_pdb_path(pdb_id: str, output_pdb_path: str = "protein.pdb") -> str:
        """
        Retrieves the PDB file from RCSB and saves it locally.
        
        Returns the path to teh saved PDB file.
        """
        pdb_id = pdb_id.strip().upper()
        
        try:
            pdbl = PDBList()
            local_file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format='pdb', overwrite=True)
            
            if not local_file_path or not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Bio.PDB failed to retrieve structure {pdb_id}.")
            
            if local_file_path != output_pdb_path:
                os.rename(local_file_path, output_pdb_path)
                local_file_path = output_pdb_path
                
            return local_file_path
        
        except Exception as e:
            raise RuntimeError(f"Error during protein retrieval for {pdb_id} : {e}")
    
    @staticmethod    
    def convert_to_pdbqt(input_path: str, output_path: str, is_receptor: bool = False) -> str:
        """
        Converts a PDB or SDF file to PDBQT using the obabel command-line tool.
        """
        command = ["obabel", input_path, "-O", output_path, "--p", "-h"]
        
        if is_receptor:
            command.append("-xr")
            
        try: 
            subprocess.run(command, check=True, capture_output=True, text=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"OpenBabel conversion failed for {input_path}. stderr: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("OpenBabel exceutable not found. Please install it or add it to your PATH.")
        pass
    
    @staticmethod    
    def generate_vina_config(center_x: float, center_y: float, center_z: float, size: float = 20.0, filename: str = "vina_config.txt") -> str:
        """
        Creates a configuration file for AutoDock Vina.
        """
        config_content = f"""
        center_x = {center_x}
        center_y = {center_y}
        center_z = {center_z}
        size_x = {size}
        size_y = {size}
        size_z = {size}
        """
        
        with open(filename, "w") as f:
            f.write(config_content)
        return filename
    
    @staticmethod
    def run_vina_docking(receptor_pdbqt_path: str, ligand_pdbqt_path: str, config_file_path: str, output_base: str = "docked") -> tuple[str, str]:
        """
        Executes the Vina docking run.
        """
        output_pdbqt = f"{output_base}_out.pdbqt"
        
        command = [
            "vina",
            "--receptor", receptor_pdbqt_path,
            "--ligand", ligand_pdbqt_path,
            "--config", config_file_path,
            "--out", output_pdbqt,
            
        ]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
            
            binding_affinity_data = result.stdout
            
            return output_pdbqt, binding_affinity_data
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Vina failed. Stderr: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Vina exceutable not found.")
        
    @staticmethod
    def calculate_ligand_center(pdb_id: str, pdb_path: str) -> tuple[float, float, float]:
        """
        Calculates the center of mass for the first HETATM (ligand) found in the PDB.
        """
        parser = PDBParser(QUIET=True)
        # The PDB path from get_protein_pdb_path is required here
        structure = parser.get_structure('protein', pdb_path)
    
        ligand_coords = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname().strip()
                    
                    if res_name == 'N3':
                        for atom in residue.get_atoms():
                            ligand_coords.append(atom.get_coord())
                            # if N3 is found, we stop and use it
                        break
                if ligand_coords: break
            if ligand_coords: break
            
        if not ligand_coords:
            for atom in structure.get_atoms():
                res_id = atom.get_full_id()[3]
                
                if res_id[0] == 'HETATM' and atom.get_parent().get_resname() != 'HOH':
                    ligand_coords.append(atom.get_coord())
                    
        if not ligand_coords:
            if pdb_id.upper() == '6LU7':
                print("INFO: N3 Ligand not found by parsing. Using manual 6LU7 coordinates.")
                return 46.0, 17.5, 49.0
            else:
                raise ValueError("No co-crystallized ligand (HETATM) found to define the binidng site.")
            
        center = np.mean(np.array(ligand_coords), axis=0)
        
        return round(center[0], 2), round(center[1], 2), round(center[2], 2)
        pass

    @staticmethod
    def clean_protein_pdb(input_pdb_path: str, output_pdb_path: str) -> str:
        """
        Removes HETATM records (like water and ions) but keeps protein ATOM records.
        """
        try:
            with open(input_pdb_path, 'r') as infile:
                lines = infile.readlines()
        
            # Filter lines: only keep ATOM records (protein)
            # We also keep TER, END, and sometimes the co-crystallized HETATM 
            # but for Vina docking receptor prep, keeping only ATOM records is the safest first step.
            cleaned_lines = [
                line for line in lines 
                if line.startswith("ATOM") or line.startswith("TER") or line.startswith("END")
            ]
        
            with open(output_pdb_path, 'w') as outfile:
                outfile.writelines(cleaned_lines)
            
            return output_pdb_path
        
        except Exception as e:
            raise RuntimeError(f"Error during PDB cleaning: {e}")
        
        
            
        
