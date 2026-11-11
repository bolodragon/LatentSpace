# LatentSpace - Differentiating protein composition in cryo-EM images

Proteins can exist in different compositions which define the biological function. Here, we investigate a hydrogenase, which is either a dimer or a tetramer but only active in the dimer form. Conventional cryo-EM workflows struggle differentiating these in the 2D projections and commonly align them together. Thus, we try to separate them according to the information we have learnt in the ML4LS course.

The main analysis platform is the jupyter notebook "expeirment.ipynb". Each cell of the notebook has a documentation of what is done and what data is needed. As input the user needs a particle stack in mrc format, a metadata file in relion star format and a pure noise dataset (from noise of the to investigated dataset, since noise is dependent on aquisition) in mrc format. 

#### TO DO: merge LatentSpace_afi for streamlit files!!! 
#### Information about the streamlit app: data output of the jupyter notebook run on 4000 images can be found in the outputs folder and are the basis for the streamlit app. Access the app here: https://latentspace.streamlit.app/


Contributions of the different team members can be found in the CONTRIBUTIONS.md

#### TO DO: software/package versions
#### TO DO: licence and citation information
#### TO DO: final tagged release / archived version
