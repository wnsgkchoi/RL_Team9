# Note  

## 2025.12.07.  

간단하게 함정은 -1, 목적지는 10으로 reward를 변경하고 돌리는 중.  
logs에 현재 agent의 state와 VLM의 output을 출력하도록 하는 중. logit과 prob의 순서는 WSEN.  
-> 이렇게 하니까 agent가 목적지로 가려 하지 않고, 벽에 계속 박는 강화학습의 가장 기본적인 문제 발생.  

### Vanilla

### Penalty

### Potential

$ R = R_{env} + \gamma \Phi(s_{t+1}) - \Phi(s_{t}) $  
$ \Phi(s) = -(\text{Manhattan Distance to Goal}) $

### Curriculum Learning  

결국 이 방식을 사용해야 하는 것인가.  

## 2025.12.08.  

### 변경점  

- num_local_env를 4로 설정할 때 생성되는 env가 공통된 golden_path를 가짐. 이는 agent가 trap을 피하고 goal에 도달해야 한다는 큰 의미의 뜻을 이해하지 못하고 해당 path만 따라가는 일종의 reward hacking을 할 가능성을 높임. (실제로 그런 것으로 보임)  
따라서 num_local_env를 8로 설정함. 이전보다는 env에 over-fit되는 문제가 완화될 것으로 보임.  
- 8로 설정해도 시작지점 바로 아래에 trap이 있는 환경이 하나라서 시작하자마자 아래로 갈 확률이 매우 높음. 이는 바로 아래 trap이 있는 환경에도 적용됨. 따라서 num_local_env를 늘리거나, 적당한 seed를 찾을 필요성이 있음. <- 그냥 계속 학습 돌리니까 해결된 것으로 보임.  

### 코드 수정  

- wandb에서 agent의 성능을 바로 알기 위해 success_count, failed_count, success_rate 등을 추가.  
    - 환경의 서로 다른 보상체계때문에 reward로는 직접적 비교가 불가능하기 때문.  
- evaluation을 여러 map에서 할 수 있도록 코드 수정  
- 