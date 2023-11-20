# CVLM

CVLM: A Multimodal VLLM

## Framework

![framework.png](assets/framework.png)

## Evaluate

### MME

```
model_path=ToOverwrite # trained model to replace  
image_path=MME_Path # MME testset path  
bash scripts/evaluation_mme.sh $model_path $image_path  

```

2023-11-18: CVLM has achieved 1636.46 perception score (No.1), 448.93 cognition score(No.2), and 2125.39 points in total (No.1). Please refer to [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)  


## Acknowledgement

[CLIP](https://github.com/openai/CLIP)  
[MiniGPT-4](https://minigpt-4.github.io/)   
[LLaVA](https://github.com/haotian-liu/LLaVA)  
[Vicuna](https://github.com/lm-sys/FastChat)  


Copyright © 2023 CVLM 

