# brain-GNN-project  

## Report
[과제연구21-1_최종보고서_20150751최기창.pdf](https://github.com/rlckd159/brain-GNN-project/files/6703237/21-1_._20150751.pdf)


##  figure
### Training Curve
![data_95_diff_base](https://user-images.githubusercontent.com/49244613/123128162-df7a3880-d485-11eb-89bc-d4c3b5d87e1d.jpg)

### Visualization 
https://brainpainter.csail.mit.edu/
![brain_painter_freq](https://user-images.githubusercontent.com/49244613/123128173-e1dc9280-d485-11eb-8233-2df97be70259.JPG)

## sparsification   

### Disparity Filtering  
> code source : http://www.michelecoscia.com/?page_id=287  

> paper : Serrano, M.A., Boguna, M., Vespignani, A.: Extracting the multiscale backbone of complex
weighted networks. Proc. Natl. Acad. Sci. 106(16), 6483–6488 (2009)  

> paper used this method for brain graphs : Xin Ma, Guorong Wu, and Won Hwa Kim. Multi-resolution graph neural network for identifying disease-
specific variations in brain connectivity. arXiv preprint arXiv:1912.01181, 2019.   


이런식으로 alpha 값을 잘 정하면, edge를 filtering 해줌.  

![image](https://user-images.githubusercontent.com/49244613/116775895-9ff42900-aaa0-11eb-8856-1408aad0d94a.png)
