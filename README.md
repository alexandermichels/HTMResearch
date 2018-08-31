# An Exploration of the Topological and Logical Properties of Hierarchical Temporal Memory Networks
---------------------------------------------------------------------------------------------------
A repository for files related to my Honors Research Thesis at Westminster College.

<center>
  <figure>
    <img src="images/hierarchy.png"  style="max-width:90%;">
    <figcaption>A Hierarchical Temporal Mememory Network</figcaption>
  </figure>
</center>

## Proposal

The Proposal for my Honors Research Thesis was due April 30th, 2018 and it is complete, feel free to check it out and ask any questions you may have.

## Thesis

The paper is ongoing!

## Ideas/Questions:

* What exactly is the output of a layer? (I think its a linear combination of the predicted elements of the input space)
  * Can I come up with a better decoding scheme? (Graph Theory/Topology?)
  * What are the output's relations to the system's predictions for t_{n+1}, t_{n+2}, ....
  * General limits of the system as a function of parameters (system architecture/"topology" such as cells/row, etc.)
* Similarity to Convolutional Neural Networks/Recurrent Neural Networks (or other existing models)
  * Cortical Learning Algorithms seem similar to CNNs (especially pooling)
  * The output (the activated neurons) seems similar to that of a RNN (hidden state)
* Can the non-binary weights be used in the system in a more effective way (rather than activation) leading to a better system (more accurate, etc.)?
  * Using fuzzy control to produce a fuzzy temporal pooler

## Implementations of Interest

* [C++ HTM](https://www.youtube.com/user/mrferrier)
* [Clortex](https://github.com/htm-community/clortex)
* [Comportex](https://github.com/htm-community/comportex)
* [HTM.java](https://github.com/numenta/htm.java)
* [NuPIC](https://github.com/numenta/nupic)


## Sources

For a comprehensive list check the **References** section of my Honors Research draft, but here is a good list to get you started:

* [Advanced NuPIC Programming](src/Advanced_NuPIC_Programming.pdf)
* [Biological and Machine Intelligence](src/Biological_and_Machine_Intelligence.pdf)
* [Encoding Data for HTM Systems](src/Encoding_Data_for_HTM_Systems.pdf)
* [Evaluation of Hierarchical Temporal Memory in algorithmic trading](src//home/alex/Documents/GitHubRepos/HonorsResearch/src/Evaluation_of_Hierarchical_Temporal_Memory_in_algorithmic_trading.pdf)
* [Hierarchical Temporal Memory including HTM Cortical Learning Algorithms](src/Hierarchical_Temporal_Memory_including_HTM_Cortical_Learning_Algorithms.pdf)
* [HTM School: Scalar Encoding](https://www.youtube.com/watch?v=V3Yqtpytif0)
* [Intelligent Predictions: an Empirical Study of the Cortical Learning Algorithm](src//home/alex/Documents/GitHubRepos/HonorsResearch/src/Intelligent_Predictions:_an_Empirical_Study_of_the_Cortical_Learning_Algorithm.pdf)
* [A Mathematical Formalization of Hierarchical Temporal Memory’s Spatial Pooler](src/A_Mathematical_Formalization_of_Hierarchical_Temporal_Memory’s_Spatial_Pooler.pdf)
* [Principles of Hierarchical Temporal Memory (HTM): Foundations of Machine Intelligence](https://www.youtube.com/watch?v=6ufPpZDmPKA)
* [Properties of Sparse Distributed Representations and their Application to Hierarchical Temporal Memory](src/Properties_of_Sparse_Distributed_Representations_and_their_Applications_to_Hierarchical_Temporal_Memory.pdf)
* [Quantum Computation via Sparse Distributed Representation](https://arxiv.org/pdf/1707.05660.pdf)
* [Random Distributed Scalar Encoder](http://fergalbyrne.github.io/rdse.html)
* [Semantic Folding: Theory and its Application in Semantic Fingerprinting](src/Semantic_Folding:_Theory_and_its_Application_in_Semantic_Fingerprinting.pdf)
* [SDR Classifier](http://hopding.com/sdr-classifier#title)
* [Towards a Mathematical Theory of Cortical Micro-circuits](src/Towards_a_Mathematical_Theory_of_Cortical_Micro-circuits.PDF)
