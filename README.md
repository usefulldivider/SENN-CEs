# SENN-CEs
# Main goal :
The main goal of this project was to address a fundamental limitation in traditional counterfactual explanation generation: they typically begin as optimization problems over the entire input feature space. However, not all input features are equally necessary for generating meaningful counterfactual explanations.
To solve this problem, we leverage Self-Explaining Neural Networks (SENN), which extract important concepts from the input data and assign them appropriate relevance scores. Using these learned concepts as a foundation, we can reduce our search space from the entire input feature set to a more focused set of relevant concepts.

# Dataset :
The dataset used is the MNIST data set.

# Code :
The code is highly annotated. There is a document string present explaining the details in almost every major parts of the code

# evaluation metrics :
For SENN classification : 
- accuracy
For counterfactual generation :
- a robustness check 
- L2 distance 
- sparsity
- concept change
- concept relevance scores
- also visual aid for comparison
# Outputs

![download](https://github.com/user-attachments/assets/946f594b-5401-4907-bcc8-1986a817a905)

Metrics Summary:<br>
Class 0→1: L2=11.69; Sparsity=55.5%; Concept Δ=-1.32<br>
Class 0→2: L2=10.48; Sparsity=42.5%; Concept Δ=-0.60<br>
Class 0→3: L2=9.16; Sparsity=35.1%; Concept Δ=-0.52<br>
Class 0→4: L2=10.00; Sparsity=41.6%; Concept Δ=-1.57<br>
Class 0→5: L2=6.03; Sparsity=22.4%; Concept Δ=-0.09<br>
Class 0→6: L2=7.69; Sparsity=32.7%; Concept Δ=-1.13<br>
Class 0→7: L2=8.24; Sparsity=35.8%; Concept Δ=-0.62<br>
Class 0→8: L2=7.89; Sparsity=36.1%; Concept Δ=-0.31<br>
Class 0→9: L2=8.28; Sparsity=35.1%; Concept Δ=-0.75<br>
![download](https://github.com/user-attachments/assets/e913a04e-768b-4dc8-856d-dad37c948088)
Metrics Summary:<br>
Class 1→0: L2=10.37; Sparsity=30.9%; Concept Δ=-0.98<br>
Class 1→2: L2=9.61; Sparsity=31.9%; Concept Δ=-0.45<br>
Class 1→3: L2=9.39; Sparsity=34.3%; Concept Δ=-0.42<br>
Class 1→4: L2=5.61; Sparsity=21.0%; Concept Δ=0.24<br>
Class 1→5: L2=8.68; Sparsity=29.8%; Concept Δ=-0.26<br>
Class 1→6: L2=9.05; Sparsity=32.0%; Concept Δ=-0.82<br>
Class 1→7: L2=7.87; Sparsity=26.8%; Concept Δ=-0.56<br>
Class 1→8: L2=8.09; Sparsity=25.5%; Concept Δ=-0.11<br>
Class 1→9: L2=6.96; Sparsity=27.6%; Concept Δ=-0.47<br>
![download](https://github.com/user-attachments/assets/65286691-2901-4cb2-86db-fcb286636b02)

Metrics Summary:<br>
Class 2→0: L2=6.65; Sparsity=24.9%; Concept Δ=-0.16<br>
Class 2→1: L2=8.63; Sparsity=42.2%; Concept Δ=-0.92<br>
Class 2→3: L2=7.03; Sparsity=31.1%; Concept Δ=0.24<br>
Class 2→4: L2=9.95; Sparsity=48.3%; Concept Δ=-1.92<br>
Class 2→5: L2=8.79; Sparsity=35.6%; Concept Δ=-0.30<br>
Class 2→6: L2=7.75; Sparsity=29.5%; Concept Δ=-0.82<br>
Class 2→7: L2=10.83; Sparsity=43.5%; Concept Δ=-1.62<br>
Class 2→8: L2=7.30; Sparsity=26.9%; Concept Δ=0.07<br>
Class 2→9: L2=9.57; Sparsity=41.3%; Concept Δ=-1.83<br>

And similar outputs for the rest of the classes.

# visualizing the Concepts 
![image](https://github.com/user-attachments/assets/365bb6b6-5b8e-4702-8c8e-6bc591c250ab)


# References : 

Alvarez-Melis & Jaakkola (2018) - Towards Robust Interpretability with Self-Explaining Neural Networks

[AmanDaVinci et al. - *Self-Explaining Neural Networks: A Review with Extensions*](https://github.com/AmanDaVinci/SENN)


