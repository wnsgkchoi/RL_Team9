# Note  

## 2025.12.07.  

간단하게 함정은 -1, 목적지는 10으로 reward를 변경하고 돌리는 중.  
logs에 현재 agent의 state와 VLM의 output을 출력하도록 하는 중. logit과 prob의 순서는 WSEN.  
-> 이렇게 하니까 agent가 목적지로 가려 하지 않고, 벽에 계속 박는 강화학습의 가장 기본적인 문제 발생.  

### Curriculum Learning  

결국 이 방식을 사용해야 하는 것인가.  


## 2025.12.08.  

### 변경점  

- num_local_env를 4로 설정할 때 생성되는 env가 공통된 golden_path를 가짐. 이는 agent가 trap을 피하고 goal에 도달해야 한다는 큰 의미의 뜻을 이해하지 못하고 해당 path만 따라가는 일종의 reward hacking을 할 가능성을 높임. (실제로 그런 것으로 보임)  
따라서 num_local_env를 8로 설정함. 이전보다는 env에 over-fit되는 문제가 완화될 것으로 보임.  

### 

1. 처음부터 8\*8으로 penalty vs 처음부터 8\*8으로 no-penalty  
2. curriculum learning으로 8\*8까지 penalty로. (no-penalty는 4\*4만. 대신 penalty가 더 좋음을 보여야 함.)  