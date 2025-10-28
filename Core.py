from google import genai
import requests
import io
from Bio.PDB import PDBParser, PDBList
from Bio.PDB.PDBExceptions import PDBConstructionException
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw
import re
import os

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
        
        '''try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                pdb_data = response.text
            else:
                # Non-200 response, try local file
                print(
                    f"Warning: RCSB returned {response.status_code} for {pdb_id}.pdb, checking local file..."
                )
        except requests.RequestException:
            # Network/connection issue, try local file
            print(
                f"Warning: Failed to fetch {pdb_id}.pdb from RCSB, checking local file..."
            )

        # Fallback: check if local file exists
        if pdb_data is None or len(pdb_data.strip()) == 0:
            local_file = f"{pdb_id}.pdb"
            if os.path.exists(local_file):
                with open(local_file, "r") as f:
                    pdb_data = f.read()
            else:
                return f"Error: Unable to fetch {pdb_id}.pdb from RCSB and no local copy found."

        if "END" not in pdb_data.splitlines()[-1]:
            return f"Error: Fetched data for {pdb_id} appears incomplete or invalid. Missing END record."
        '''
        
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
    def __init__(self, api_key: str):
        """
        Initialize the agent and connect to OpenAI's API

        Args:
            api_key: OpenAI API key
        """
        self.TEMPERATURE = 0.7
        self.MAX_TOKENS = 10000
        self.MODEL = "gemini-2.5-flash"
        self.PDB_BASE = "https://files.rcsb.org/download/"
        self.SYSTEM = {
            "role": "system",
            "content": "You are an experienced drug-designer who specializes in structure based drud-design.",
        }

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

    def reopen_client(self):
        self.client = genai.Client(api_key=self.api_key)

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

        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                system_instruction=self.SYSTEM["content"],
                max_output_tokens=self.MAX_TOKENS,
            )
        )

        result = {
            "summary": protein_summary,
            "analysis": strip_md(response.text),
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

        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                system_instruction=self.SYSTEM["content"],
                max_output_tokens=self.MAX_TOKENS,
            )
        )

        result = response.text
        return strip_md(result)

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

        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(
                temperature=self.TEMPERATURE,
                system_instruction=self.SYSTEM["content"],
                max_output_tokens=self.MAX_TOKENS,
            )
        )

        result = response.text
        return strip_md(result)
