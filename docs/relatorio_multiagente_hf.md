**Relatório Técnico**

**Resumo da Deteção**
O presente relatório apresenta os resultados obtidos através do uso de modelos de Machine Learning (MLP Scikit-Learn) e Deep Learning (TensorFlow) aplicados ao dataset HTRU2, que abrange 100 sinais de estrelas de neutrões e pulsares. Os modelos foram treinados para identificar padrões únicos em cada sinal, permitindo uma classificação precisa entre pulsares confirmados e candidatos potenciais.

Os principais candidatos detectados são:

1. **Sinal #35**: Este sinal tem uma probabilidade de ser um pulsar de ML de 99,99% e de TensorFlow de 100%. O seu indicador de média IP (-3,57) e o excedente de curtose IP (5,31) são altamente significativos, sugerindo uma forte presença de pulsos.

2. **Sinal #61**: Com uma probabilidade de ML de 99,82% e de TensorFlow de 100%, este sinal também é altamente suspeito de ser um pulsar. A sua média IP (-3,45) e o excedente de curtose IP (3,84) são ainda mais evidentes, apontando para uma configuração de pulsos.

3. **Sinal #64**: Este sinal tem uma probabilidade de ML de 99,40% e de TensorFlow de 99,97%. A sua média IP (-2,47) e o ex