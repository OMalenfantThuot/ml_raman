import numpy as np
from mlcalcdriver.workflows.phonon import  DummyCalculator, PhononFromHessian
from scipy.sparse.linalg import eigsh
from mlcalcdriver import Job, Posinp
from mlcalcdriver.globals import HA_TO_CMM1
from ml_raman.raman.gamma_modes_pristine import gamma_modes_pristine
class PhononFromHessianLanczos(PhononFromHessian):
    def __init__(self, posinp, hessian, diag_mode, neigs, initial_guess):
        super().__init__(
            posinp=posinp,
            hessian=hessian
        )
        self.diag_mode = diag_mode
        self.neigs = neigs
        self.initial_guess = initial_guess

    def _solve_dyn_mat_exact(self):
        r"""
        Obtains the eigenvalues and eigenvectors from
        the dynamical matrix
        """
        eigs, vecs = np.linalg.eigh(self.dyn_mat)
        eigs = np.sign(eigs) * np.sqrt(np.where(eigs < 0, -eigs, eigs))
        return eigs, vecs

    def _solve_dyn_mat_lanczos(self):
        r"""
        Obtains the neigs highest eigenvalues and corresponding eigenvectors from
        the dynamical matrix
        """
        print('length of v1 and v2 vector', self.dyn_mat.shape[0])
        
        v1, v2 = gamma_modes_pristine(int(self.dyn_mat.shape[0]/3))
        print('type of v1 and v2 vector', type(v1))
        print(v1)
        if self.initial_guess == 1:
            eigs, vecs = eigsh(self.dyn_mat, k=self.neigs, v0=np.array(v1))
        elif self.initial_guess == 2:
            eigs, vecs = eigsh(self.dyn_mat, k=self.neigs, v0=v2)
        else:    
            eigs, vecs = eigsh(self.dyn_mat, k=self.neigs)
        eigs = np.sign(eigs) * np.sqrt(np.where(eigs < 0, -eigs, eigs))
        return eigs, vecs
     
    def _post_proc_exact_lanczos(self, job):
        r"""
        Calculates the energies and normal modes from the results
        obtained from the model.
        """
        self.dyn_mat = self._compute_dyn_mat(job)
        if self.diag_mode =="exact":
            self.energies, self.normal_modes = self._solve_dyn_mat_exact()
            self.energies *= HA_TO_CMM1
        elif self.diag_mode =="lanczos":
            self.energies, self.normal_modes = self._solve_dyn_mat_lanczos()
            self.energies *= HA_TO_CMM1
        else:
            raise NotImplementedError()

    def run_exact_lanczos(self):
        job = Job(posinp=self.posinp, calculator=self.calculator)
        job.results["hessian"] = self.hessian
        self._post_proc_exact_lanczos(job)

