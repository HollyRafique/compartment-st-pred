import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional
from sklearn.base import RegressorMixin

from dataclasses import dataclass
#from typing import Dict, Tuple, List, Any, Iterable

@dataclass(frozen=True)
class Calibration:
    alphas: np.ndarray
    betas: np.ndarray
    genes: Iterable[str]

    def __post_init__(self):
        if len(self.alphas) != len(self.betas):
            raise ValueError("alphas and betas must have same length")
        if len(self.alphas) != len(self.genes):
            raise ValueError("alphas/betas must match number of genes")

    def merge(self, other: "Calibration", overwrite_dups=True) -> "Calibration":
        if set(self.genes) & set(other.genes):
            if overwrite_dups:
                print("warning, overlapping genes in merge sets - overwriting the original")
            else:
                raise ValueError("Overlapping genes in calibration merge.")

        # ---- Remove overlapping genes from self ----
        keep_mask_self = [g not in other.genes for g in self.genes]

        filtered_self_genes = [g for g, k in zip(self.genes, keep_mask_self) if k]
        filtered_self_alpha = self.alphas[keep_mask_self]
        filtered_self_beta  = self.betas[keep_mask_self]

        # ---- Combine: filtered self first, then full other ----
        new_genes = filtered_self_genes + other.genes
        new_alpha = np.concatenate([filtered_self_alpha, other.alphas])
        new_beta  = np.concatenate([filtered_self_beta, other.betas])

        return Calibration(
            alphas=new_alpha,
            betas=new_beta,
            genes=new_genes
        )


    def apply(self, y_pred):
        """
        Apply affine calibration to predictions.

        y_pred:
          - np.ndarray of shape (n_samples, n_genes), OR
          - pd.DataFrame with gene columns matching self.genes

        Returns:
          calibrated predictions, same type as input
        """
        if isinstance(y_pred, pd.DataFrame):
            genes = list(y_pred.columns)
        
            if set(self.genes) != set(genes):
                raise ValueError(
                    "Gene mismatch between predictions and calibration."
                )
            ### instead of raising an error just fix it
            ### order incoming as per self.genes
            y_pred = y_pred[self.genes]

            y = y_pred.to_numpy()
            return_df = True



        else:
            ## blindly assumes it is in the right order with the right genes
            genes = None
            y = np.asarray(y_pred)
            return_df = False

        if y.shape[1] != len(self.alphas):
            raise ValueError(
                f"Prediction has {y.shape[1]} genes but calibration has {len(self.alphas)}."
            )


            
            


        y_cal = y * self.alphas + self.betas

        bad = ~np.isfinite(y_cal)
        if np.any(bad):
            y_cal = y_cal.copy()
            y_cal[bad] = y[bad]

        if return_df:
            return pd.DataFrame(y_cal, columns=genes, index=y_pred.index)

        return y_cal


    def to_dataframe(self) -> pd.DataFrame:
        """Return calibration as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "Gene": list(self.genes),
                "alpha": self.alphas,
                "beta": self.betas,
            }
        )

    def to_csv(self, path: str, **kwargs):
        """Save calibration to CSV."""
        self.to_dataframe().to_csv(path, index=False, **kwargs)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Reconstruct Calibration from a DataFrame."""
        required = {"Gene", "alpha", "beta"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return cls(
            alphas=df["alpha"].to_numpy(),
            betas=df["beta"].to_numpy(),
            genes=df["Gene"].tolist(),
        )

    @classmethod
    def from_csv(cls, path: str, **kwargs):
        """Load calibration from CSV."""
        df = pd.read_csv(path, **kwargs)
        return cls.from_dataframe(df)



    

class GeneModelStore:
    """
    Manage a collection of trained regression models keyed by gene name.

    - Save/load models to/from disk (one file per gene).
    - Run predictions for one or many genes.
    """

    def __init__(self,
                 models: Optional[Dict[str, RegressorMixin]] = None,
                 calibration: Optional[Calibration] = None,
                 model_dir: str = "models_by_gene",
                 file_ext: str = ".joblib"):
        self.models: Dict[str, RegressorMixin] = models or {}
        
        self.calibration = calibration
        self.model_dir = model_dir
        self.file_ext = file_ext
        os.makedirs(self.model_dir, exist_ok=True)

    # --------- basic info ---------
    @property
    def genes(self) -> Iterable[str]:
        return list(self.models.keys())

    # --------- saving / loading ---------
    def _validate_calibration_alignment(self) -> None:
        if self.calibration is None:
            return

        model_genes = sorted(list(self.models.keys()))
        calib_genes = sorted(list(self.calibration.genes))

        if set(model_genes) != set(calib_genes):
            raise ValueError(
                "Model genes and calibration genes do not match."
            )

        if model_genes != calib_genes:
            raise ValueError(
                "Gene order mismatch between models and calibration."
            )



    def _model_path(self, gene: str) -> str:
        # You can add cleaning of gene names here if needed
        return os.path.join(self.model_dir, f"{gene}{self.file_ext}")

    def save_all(self) -> None:
        """
        Save all models in `self.models` to disk (one file per gene).
        """
        self._validate_calibration_alignment()

        for gene, model in self.models.items():
            path = self._model_path(gene)
            joblib.dump(model, path)

        # optional: save gene index
        index_path = os.path.join(self.model_dir, "genes.txt")
        with open(index_path, "w") as f:
            for gene in self.genes:
                f.write(f"{gene}\n")
                
        # calibration
        if self.calibration is not None:
            calib_path = os.path.join(self.model_dir, "calibration.csv")
            self.calibration.to_csv(calib_path)


    def load_all(self) -> None:
        """
        Load all models found in `model_dir` into `self.models`.
        Assumes one file per gene.
        """
        models = {}
        for fname in os.listdir(self.model_dir):
            if not fname.endswith(self.file_ext):
                continue
            gene = fname[: -len(self.file_ext)]
            path = os.path.join(self.model_dir, fname)
            models[gene] = joblib.load(path)
        self.models = models
            # load calibration if present
        calib_path = os.path.join(self.model_dir, "calibration.csv")
        if os.path.exists(calib_path):
            self.calibration = Calibration.from_csv(calib_path)
        else:
            self.calibration = None

    def load_selected(self, genes: Iterable[str]) -> None:
        """
        Load only a subset of genes from disk into `self.models`,
        overwriting any existing entries for those genes.
        """
        for gene in genes:
            path = self._model_path(gene)
            if not os.path.exists(path):
                raise FileNotFoundError(f"No saved model for gene {gene!r} at {path}")
            self.models[gene] = joblib.load(path)

    # --------- prediction helpers ---------
    def predict_gene(self, gene: str, X) -> np.ndarray:
        """
        Predict expression for a single gene.
        """
        if gene not in self.models:
            raise KeyError(f"Model for gene {gene!r} not loaded.")
        return self.models[gene].predict(X)

    def predict_many(self,
                     genes: Optional[Iterable[str]],
                     X,
                     as_dataframe: bool = True,
                     row_names=None,
                     apply_calibration: bool = True) -> pd.DataFrame | np.ndarray:
        """
        Predict expression for many genes at once.

        Parameters
        ----------
        genes : iterable of gene names, or None
            If None, use all loaded genes.
        X : array-like of shape (n_samples, n_features)
        as_dataframe : bool
            If True, return a DataFrame (samples × genes).
            Else, return a 2D NumPy array with columns in the same order as `genes`.
        """
        if genes is None:
            genes = list(self.genes)
        else:
            genes = list(genes)

        preds = []
        for i, gene in enumerate(genes):
            print(f"{i}: {gene}")
            if gene not in self.models:
                raise KeyError(f"Model for gene {gene!r} not loaded.")
            y_pred = self.models[gene].predict(X)
            preds.append(y_pred)

        preds = np.vstack(preds).T  # shape: (n_samples, n_genes)

        # ---- calibration ----
        if apply_calibration and self.calibration is not None:
            # enforce gene alignment
            if list(self.calibration.genes) != genes:
                raise ValueError(
                    "Gene order mismatch between predictions and calibration."
                )
            preds = self.calibration.apply(preds)

        if as_dataframe:
            df = pd.DataFrame(preds, columns=genes)

            if row_names:
                df.index = row_names
            return df
        
        return preds
