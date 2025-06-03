
import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
import requests
import json
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ddi_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class DrugDataCollector:
    """Collects and processes drug data from multiple sources"""
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.chembl_url = "https://www.ebi.ac.uk/chembl/api/data"
        logger.info(f"Initialized DrugDataCollector with cache_dir: {cache_dir}")
    
    def _get_from_cache_or_download(self, filename: str, download_func) -> pd.DataFrame:
        """Get data from cache if available, otherwise download"""
        cache_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(cache_path):
            logger.info(f"Loading {filename} from cache")
            return pd.read_csv(cache_path)
            
        logger.info(f"Downloading {filename}")
        df = download_func()
        df.to_csv(cache_path, index=False)
        return df
    
    def get_drugs(self, limit: int = 1000) -> pd.DataFrame:
        """Get approved drugs from ChEMBL"""
        def download_drugs():
            drugs_data = []
            offset = 0
            
            while len(drugs_data) < limit:
                url = f"{self.chembl_url}/molecule?format=json&max_phase=4&limit=100&offset={offset}"
                response = requests.get(url)
                response.raise_for_status()
                
                molecules = response.json()['molecules']
                if not molecules:
                    break
                    
                for mol in molecules:
                    if 'molecule_structures' in mol and mol['molecule_structures']:
                        drugs_data.append({
                            'drug_id': mol['molecule_chembl_id'],
                            'name': mol['pref_name'] or mol['molecule_chembl_id'],
                            'smiles': mol['molecule_structures']['canonical_smiles'],
                            'inchi': mol['molecule_structures']['standard_inchi'],
                            'formula': mol['molecule_properties']['full_molformula']
                        })
                
                offset += 100
                
            return pd.DataFrame(drugs_data)
        
        return self._get_from_cache_or_download('drugs.csv', download_drugs)
    
    def get_interactions(self, drug_ids: List[str]) -> pd.DataFrame:
        """Get drug interactions using structural similarity"""
        def download_interactions():
            interactions = []
            mol_dict = {}  # Cache for molecules
            
            # Calculate structural similarity based interactions
            for i, drug1 in enumerate(drug_ids):
                for drug2 in drug_ids[i+1:]:
                    try:
                        # Get drug structures
                        if drug1 not in mol_dict:
                            drug1_data = self.get_drugs()[self.get_drugs()['drug_id'] == drug1].iloc[0]
                            mol1 = Chem.MolFromSmiles(drug1_data['smiles'])
                            mol_dict[drug1] = mol1
                        else:
                            mol1 = mol_dict[drug1]
                            
                        if drug2 not in mol_dict:
                            drug2_data = self.get_drugs()[self.get_drugs()['drug_id'] == drug2].iloc[0]
                            mol2 = Chem.MolFromSmiles(drug2_data['smiles'])
                            mol_dict[drug2] = mol2
                        else:
                            mol2 = mol_dict[drug2]
                        
                        # Calculate structural similarity
                        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
                        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
                        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                        
                        # Create realistic interaction based on similarity
                        interaction = 1 if similarity > 0.5 else 0
                        severity = 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                        
                        mechanism = np.random.choice([
                            'CYP3A4 inhibition',
                            'P-glycoprotein interference',
                            'Additive effects',
                            'pH alteration'
                        ]) if interaction == 1 else 'none'
                        
                        interactions.append({
                            'drug_1': drug1,
                            'drug_2': drug2,
                            'interaction': interaction,
                            'severity': severity,
                            'mechanism': mechanism,
                            'similarity': similarity
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing drug pair {drug1}-{drug2}: {str(e)}")
                        continue
            
            interactions_df = pd.DataFrame(interactions)
            # Balance the dataset
            pos_samples = interactions_df[interactions_df['interaction'] == 1]
            neg_samples = interactions_df[interactions_df['interaction'] == 0].sample(
                n=len(pos_samples), random_state=42)
            balanced_df = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)
            
            return balanced_df
        
        return self._get_from_cache_or_download('interactions.csv', download_interactions)

class MolecularFeatureExtractor:
    def __init__(self, n_bits: int = 2048):
        logger.info(f"Initializing MolecularFeatureExtractor with {n_bits} bits")
        self.n_bits = n_bits
        self.failed_smiles = []
    
    def extract_features(self, smiles: str) -> Optional[np.ndarray]:
        logger.debug(f"Processing SMILES: {smiles[:50]}...")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                self.failed_smiles.append(smiles)
                return None
            
            logger.debug("Generating molecular fingerprints")
            morgan = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.n_bits)))
            maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)))
            rdkit = np.array(list(Chem.RDKFingerprint(mol)))
            
            logger.debug("Calculating molecular descriptors")
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumAliphaticCarbocycles(mol),
                Descriptors.NumAliphaticHeterocycles(mol),
                Descriptors.NumAromaticCarbocycles(mol),
                Descriptors.NumAromaticHeterocycles(mol),
                Descriptors.NumHeteroatoms(mol)
            ]
            
            features = np.concatenate([morgan, maccs, rdkit, descriptors])
            logger.debug(f"Generated {len(features)} features successfully")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(f"Failed SMILES: {smiles}")
            self.failed_smiles.append(smiles)
            return None

class DDIModel(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super(DDIModel, self).__init__()
        logger.info("Initializing DDI Model with configuration:")
        logger.info(f"- Input features: {num_features}")
        logger.info(f"- Hidden dimension: {hidden_dim}")
        logger.info(f"- Attention heads: {num_heads}")
        logger.info(f"- Dropout rate: {dropout}")
        
        # Three GAT layers
        self.conv1 = GATv2Conv(
            num_features,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        
        self.conv2 = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        
        self.conv3 = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.ln1 = torch.nn.LayerNorm(hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)
        self.ln3 = torch.nn.LayerNorm(hidden_dim)
        
        # MLP for prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info("Model architecture initialized successfully")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First GAT layer
        logger.debug(f"Forward pass - Input shape: {x.shape}")
        x1 = self.conv1(x, edge_index)
        x1 = self.ln1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)
        logger.debug(f"Layer 1 output shape: {x1.shape}")
        
        # Second GAT layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.ln2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.3, training=self.training)
        x2 = x2 + x1  # Skip connection
        logger.debug(f"Layer 2 output shape: {x2.shape}")
        
        # Third GAT layer
        x3 = self.conv3(x2, edge_index)
        x3 = self.ln3(x3)
        x3 = F.elu(x3)
        x3 = x3 + x2  # Skip connection
        logger.debug(f"Layer 3 output shape: {x3.shape}")
        
        return x3
    
    def predict_pair(
        self,
        embeddings: torch.Tensor,
        idx1: int,
        idx2: int
    ) -> torch.Tensor:
        logger.debug(f"Predicting interaction for indices {idx1}, {idx2}")
        pair_embedding = torch.cat([embeddings[idx1], embeddings[idx2]])
        return torch.sigmoid(self.mlp(pair_embedding))

class DDIPredictor:
    """Main class for DDI prediction with optimized prediction capabilities"""
    
    def __init__(
        self,
        model_dir: str = "models",
        data_dir: str = "data",
        device: str = None
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.drug_to_idx = None
        self.idx_to_drug = None
        self.cached_data = None
        self.cached_embeddings = None
        self.feature_extractor = MolecularFeatureExtractor()
        self.data_collector = DrugDataCollector(data_dir)
    
    def prepare_data(self) -> Tuple[Data, pd.DataFrame, pd.DataFrame]:
        logger.info("Starting data preparation")
        
        logger.info("Step 1: Loading drug data")
        drugs_df = self.data_collector.get_drugs()
        logger.info(f"Loaded {len(drugs_df)} drugs")
        
        logger.info("Step 2: Extracting molecular features")
        features = []
        valid_indices = []
        
        for idx, row in tqdm(drugs_df.iterrows(), desc="Extracting features"):
            feature_vector = self.feature_extractor.extract_features(row['smiles'])
            if feature_vector is not None:
                features.append(feature_vector)
                valid_indices.append(idx)
        
        drugs_df = drugs_df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Successfully processed {len(drugs_df)} drugs")
        
        self.drug_to_idx = {
            drug_id: idx for idx, drug_id in enumerate(drugs_df['drug_id'])
        }
        self.idx_to_drug = {
            idx: drug_id for drug_id, idx in self.drug_to_idx.items()
        }
        
        logger.info("Step 3: Getting drug interactions")
        interactions_df = self.data_collector.get_interactions(
            drugs_df['drug_id'].tolist())
        logger.info(f"Loaded {len(interactions_df)} interactions")
        
        logger.info("Step 4: Creating graph data")
        x = torch.FloatTensor(np.array(features))
        edge_index = []
        edge_attr = []
        
        for _, row in interactions_df.iterrows():
            if row['interaction'] == 1:
                idx1 = self.drug_to_idx[row['drug_1']]
                idx2 = self.drug_to_idx[row['drug_2']]
                
                edge_index.extend([[idx1, idx2], [idx2, idx1]])
                
                severity_val = {'high': 1.0, 'medium': 0.5, 'low': 0.1}[row['severity']]
                edge_attr.extend([severity_val, severity_val])
        
        edge_index = torch.LongTensor(edge_index).t()
        edge_attr = torch.FloatTensor(edge_attr)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        logger.info("Data preparation completed")
        
        return data, drugs_df, interactions_df
    
    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.0005,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        logger.info("Starting model training")
        data, drugs_df, interactions_df = self.prepare_data()
        
        # Store the data for future use
        self.cached_data = data
        
        logger.info("Splitting data into train/validation sets")
        train_idx, val_idx = train_test_split(
            range(len(interactions_df)),
            test_size=validation_split,
            stratify=interactions_df['interaction'],
            random_state=42
        )
        logger.info(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")
        
        logger.info("Initializing model")
        self.model = DDIModel(
            num_features=data.x.size(1),
            hidden_dim=512,
            num_heads=8,
            dropout=0.3
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
        
        best_val_auc = 0
        patience = 90
        patience_counter = 0
        
        logger.info("Starting training loop")
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            train_pbar = tqdm(range(0, len(train_idx), batch_size),
                           desc=f'Epoch {epoch+1}/{n_epochs}',
                           total=len(train_idx)//batch_size)
            
            for i in train_pbar:
                batch_idx = train_idx[i:i + batch_size]
                batch_interactions = interactions_df.iloc[batch_idx]
                
                optimizer.zero_grad()
                
                node_embeddings = self.model(
                    data.x.to(self.device),
                    data.edge_index.to(self.device)
                )
                
                predictions = []
                labels = []
                
                for _, row in batch_interactions.iterrows():
                    idx1 = self.drug_to_idx[row['drug_1']]
                    idx2 = self.drug_to_idx[row['drug_2']]
                    
                    pred = self.model.predict_pair(
                        node_embeddings, idx1, idx2)
                    predictions.append(pred)
                    labels.append(row['interaction'])
                
                predictions = torch.stack(predictions)
                labels = torch.FloatTensor(labels).to(self.device)
                
                loss = F.binary_cross_entropy(predictions.squeeze(), labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{(total_loss/batch_count):.4f}'
                })
            
            avg_train_loss = total_loss / batch_count
            history['train_loss'].append(avg_train_loss)
            
            val_metrics = self._validate(data, interactions_df.iloc[val_idx])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['auc'])
            
            scheduler.step(val_metrics['auc'])
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                self.save_model()
                logger.info(f"New best AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            logger.info(
                f"Epoch {epoch+1} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )
        
        # Generate and cache embeddings for future predictions
        self.model.eval()
        with torch.no_grad():
            self.cached_embeddings = self.model(
                data.x.to(self.device),
                data.edge_index.to(self.device)
            )
        
        return history
    
    def _validate(
        self,
        data: Data,
        val_interactions: pd.DataFrame
    ) -> Dict[str, float]:
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            node_embeddings = self.model(
                data.x.to(self.device),
                data.edge_index.to(self.device)
            )
            
            for _, row in val_interactions.iterrows():
                idx1 = self.drug_to_idx[row['drug_1']]
                idx2 = self.drug_to_idx[row['drug_2']]
                
                pred = self.model.predict_pair(
                    node_embeddings, idx1, idx2)
                predictions.append(pred.item())
                labels.append(row['interaction'])
            
            predictions_tensor = torch.FloatTensor(predictions).to(self.device)
            labels_tensor = torch.FloatTensor(labels).to(self.device)
            
            loss = F.binary_cross_entropy(predictions_tensor, labels_tensor)
            auc = roc_auc_score(labels, predictions)
            
            return {
                'loss': loss.item(),
                'auc': auc
            }
    
    def predict(
        self,
        drug1_id: str,
        drug2_id: str,
        return_details: bool = False
    ) -> Dict[str, Union[float, bool, str]]:
        
        if self.model is None:
            logger.error("Model not trained or loaded")
            raise ValueError("Model not trained or loaded")
        
        try:
            # Load the drugs data for metadata only if needed
            drugs_df = self.data_collector.get_drugs() if return_details else None
            
            # Check if drugs exist in the mapping
            if drug1_id not in self.drug_to_idx or drug2_id not in self.drug_to_idx:
                missing = []
                if drug1_id not in self.drug_to_idx:
                    missing.append(drug1_id)
                if drug2_id not in self.drug_to_idx:
                    missing.append(drug2_id)
                raise KeyError(f"Drug(s) not found: {', '.join(missing)}")
            
            idx1 = self.drug_to_idx[drug1_id]
            idx2 = self.drug_to_idx[drug2_id]
            
            # Use cached embeddings if available, otherwise generate them
            if self.cached_embeddings is None:
                logger.info("Generating node embeddings for prediction")
                
                if self.cached_data is None:
                    logger.info("No cached data found, preparing data")
                    self.cached_data, _, _ = self.prepare_data()
                
                self.model.eval()
                with torch.no_grad():
                    self.cached_embeddings = self.model(
                        self.cached_data.x.to(self.device),
                        self.cached_data.edge_index.to(self.device)
                    )
            
            # Use the cached embeddings
            self.model.eval()
            with torch.no_grad():
                prob = self.model.predict_pair(
                    self.cached_embeddings, idx1, idx2).item()
            
            logger.info(f"Predicted interaction probability: {prob:.4f}")
            
            result = {
                "probability": prob,
                "interaction_predicted": prob > 0.5,
                "confidence": "high" if abs(prob - 0.5) > 0.3 else 
                            "medium" if abs(prob - 0.5) > 0.1 else "low"
            }
            
            if return_details and drugs_df is not None:
                drug1_info = drugs_df[drugs_df['drug_id'] == drug1_id].iloc[0]
                drug2_info = drugs_df[drugs_df['drug_id'] == drug2_id].iloc[0]
                
                result.update({
                    "drug1_name": drug1_info['name'],
                    "drug2_name": drug2_info['name'],
                    "drug1_formula": drug1_info['formula'],
                    "drug2_formula": drug2_info['formula']
                })
            
            return result
                
        except KeyError as e:
            logger.error(f"Drug not found: {str(e)}")
            return {"error": f"Drug not found: {str(e)}"}
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.model_dir, f"ddi_model_{timestamp}.pt")
        torch.save(self.model.state_dict(), model_path)
        
        mappings_path = os.path.join(self.model_dir, f"mappings_{timestamp}.pkl")
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'drug_to_idx': self.drug_to_idx,
                'idx_to_drug': self.idx_to_drug
            }, f)
        
        # Save the cached data if it exists
        if hasattr(self, 'cached_data') and self.cached_data is not None:
            data_path = os.path.join(self.model_dir, f"cached_data_{timestamp}.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump(self.cached_data, f)
        
        logger.info(f"Model and mappings saved at timestamp {timestamp}")
        return timestamp
    
    def load_model(self, timestamp: str):
        model_path = os.path.join(self.model_dir, f"ddi_model_{timestamp}.pt")
        mappings_path = os.path.join(self.model_dir, f"mappings_{timestamp}.pkl")
        data_path = os.path.join(self.model_dir, f"cached_data_{timestamp}.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(mappings_path):
            logger.error(f"Model files not found for timestamp {timestamp}")
            raise ValueError("Model files not found")
        
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.drug_to_idx = mappings['drug_to_idx']
            self.idx_to_drug = mappings['idx_to_drug']
        
        # Load cached data if available
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                self.cached_data = pickle.load(f)
            logger.info("Cached data loaded successfully")
        else:
            # If cached data not available, prepare it
            self.cached_data, _, _ = self.prepare_data()
            logger.info("Cached data not found, prepared new data")
        
        # Initialize the model
        self.model = DDIModel(num_features=self.cached_data.x.size(1)).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        
        # Generate embeddings once for prediction
        self.model.eval()
        with torch.no_grad():
            self.cached_embeddings = self.model(
                self.cached_data.x.to(self.device),
                self.cached_data.edge_index.to(self.device)
            )
        
        logger.info(f"Model and mappings loaded from timestamp {timestamp}")

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Validation AUC', color='green')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Good Performance (0.8)')
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def main():
    logger.info("Starting DDI prediction pipeline")
    try:
        predictor = DDIPredictor()
        
        logger.info("Training model")
        history = predictor.train(
            n_epochs=100,
            batch_size=64,
            learning_rate=0.0005
        )
        
        # Plot and save training history
        plot_training_history(history, save_path="training_history.png")
        
        drug_pairs = [
            ("CHEMBL25", "CHEMBL2"),
            ("CHEMBL25", "CHEMBL3"),
            ("CHEMBL2", "CHEMBL3")
        ]
        
        logger.info("Making predictions")
        for drug1_id, drug2_id in drug_pairs:
            result = predictor.predict(drug1_id, drug2_id, return_details=True)
            print(f"\nPrediction for {result.get('drug1_name', drug1_id)} - "
                  f"{result.get('drug2_name', drug2_id)}:")
            for k, v in result.items():
                print(f"{k}: {v}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()