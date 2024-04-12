
# GradiEnt (versão para qualificação)

Script para análise fonológica de variáveis fonéticas contínuas. Desenvolvido em 2024 durante minha pesquisa de pós-graduação. A versão apresentada aqui foi concluída entre março e abril de 2024, com intenção de apresentá-la para qualificação do Mestrado.

# Arquivos
Este repositório contém 3 arquivos de importância:

### GradiEnt.py:

Script para rodar as análises. Basta rodar no terminal com as amostras na mesma pasta, e.g.:

> $ python3 GradiEnt.py

Para selecionar os parâmetros, modifique a seguinte seção do código:

>        'resolucao':1000,
>        'alfa': 0.0065, 
>        'beta': 0.01,   
>        'alvo_F1': 421,  # ALVO F1 DA FUNÇÃO DA RESTRIÇÃO ARTICULATÓRIA
>        'alvo_F2': 1887, # ALVO F2 DA FUNÇÃO DA RESTRIÇÃO ARTICULATÓRIA
>        'limiar_1': 600,    # F1 LIMIAR DA FUNÇÃO DA RESTRIÇÃO PERCEPTUAL
>        'limiar_2': 345,    # F1 LIMIAR DA FUNÇÃO DA RESTRIÇÃO PERCEPTUAL
>        'neutro_F1': 610,
>        'neutro_F2': 1900,
>        'dur': 100,  
>        'L': 1,    # VARIÁVEL L (TETO DAS VIOLAÇÕES) DA RESTRIÇÃO PERCEPTUAL
>        'k_1': 1,    # VARIÁVEL K (INCLINAÇÃO DO LIMIAR 1) DA RESTRIÇÃO PERCEPTUAL
>        'k_2': 7,    # VARIÁVEL K (INCLINAÇÃO DO LIMIAR 2) DA RESTRIÇÃO PERCEPTUAL
>        'gamma_rp': 0.1,  
>        'a_F1': 450,    # LIMITE INFERIOR DA INTEGRAL DE F1 (MIN. - CANDIDATO F1)
>        'b_F1': 543,    # LIMITE SUPERIOR DA INTEGRAL DE F1 (MAX. - CANDIDATO F1)
>        'a_F2': 1700,   # LIMITE INFERIOR DA INTEGRAL DE F2 (MIN. - CANDIDATO F2)
>        'b_F2': 1987,   # LIMITE SUPERIOR DA INTEGRAL DE F1 (MAX. - CANDIDATO F2)
>        'min_F1': candidatos['F*1'].min(),   
>        'max_F1': candidatos['F*1'].max(),     
>        'min_F2': candidatos['F*2'].min(),    
>        'max_F2': candidatos['F*2'].max(),   
>        'a_dur': 90,   
>        'b_dur': 110 

A combinação de *a_F1, b_F1; a_F2, b_F2* é que compõe o candidato contínuo (F1,F2) da análise. No mais, caso você queira achar os pesos para seus próprios dados, você precisa desabilitar os pesos atuais (basta colocar um # na frente de cada linha):

>        lambda_zero = 1.0277987671182454
>        lambda_RA = 0.018088342454995333
>        lambda_RP = 0.41709655507658977

E, então, habilitar (descomentar, remover o # inicial das linhas) a seguinte seção:

>        #otimizacao = minimize(otimizador.funcao_objetivo, lambdas_iniciais, method='L-BFGS-B', bounds=limites, options={'maxiter': 1000})
>        #lambdas_otimizados = otimizacao.x
>        #lambda_zero = lambdas_otimizados[0]
>        #lambda_RA = lambdas_otimizados[1]
>        #lambda_RP = lambdas_otimizados[2]

Isso faz com que o script rode o comando *minimize()* que busca os pesos ótimos que minimizam a divergência de Kullback-Leibler entre distribuição de máxima entropia e a distribuição estimada dos dados reais. No entanto, como isso é um procedimento que pode levar algum tempo, não é interessante manter a otimização permanentemente habilitada. Após você identificar os pesos das restrições para seus dados, basta desabilitar a otimização e habilitar novamente os pesos agora com seus valores:

>        lambda_zero = [substituir aqui pelo peso]
>        lambda_RA = [substituir aqui pelo peso]
>        lambda_RP = [substituir aqui pelo peso]

Assim, você pode rodar o algoritmo algumas vezes, ver o valor dos pesos das restrições e depois desabilitar a otimização. Basta preencher os pesos manualmente. Isso te salva tempo, porque agora você pode rodar várias vezes para testar combinações de candidatos diferentes sem precisar passar pela etapa de otimização todas as vezes. É importante notar que isso só é possível porque a otimização não depende dos candidatos, ela depende dos dados em geral -- então, os pesos das restrições variam conforme o *set* de dados. Isso significa que só precisamos encontrar os pesos uma única vez (ou algumas, se formos bem rigorosos) por *set* de dados. 

### amostras.txt:

Arquivo que contém os valores dos formantes (F1 e F2) analisados. Para entender melhor sua formatação, basta ler este arquivo e comparar com a função *ler_dados()* do GradiEnt, que filtra elementos pelo cabeçalho contendo palavras-chave. Estando em formatação semelhante, você pode usar outros dados/arquivos.

### GradiEnt_comentado.py:

Versão mais completa do GradiEnt que gera uma grande quantidade de *plots* (alguns deles para *debugging* e muitos deles inúteis para uma análise, só servindo para apoio visual durante minha escrita da qualificação) e está amplamente comentado. Logo, é consideravelmente mais didático se sua intenção for compreender o que está embaixo do capô.

# Equações

As principais equações que esse script roda são a restrição perceptual:

$$ R_{P}(F1) = L - \frac{L^2}{(1 + e^{k_1(F1 - F1_{\text{limiar}{1}})})(1 + e^{-k_2(F1 - F1_{\text{limiar}_{2}})})} $$

A restrição articulatória:

$$R_A(F1,F2) = \sqrt{(G_{F1} - A_{F1})^2 + (G_{F2} - A_{F2})^2} \cdot \frac{E_{\texttt{realizado}}}{E_{\texttt{esperado}}}$$

O cômputo de violações:

$$v_i(x) = e^{R_i(x)-\hat{f}(x)}$$

O cálculo do escore harmônico:

$$\mathcal{H}(x) = \sum_{i=1}^{n} p_i \cdot v_i(x)$$
A estabilidade do escore harmônico:

$$\mathcal{E} = \left( \prod_{i=1}^{n} (b_i - a_i) \right)^{-1} \cdot \int_{a_i}^{b_i} \left( \sum_{i=1}^{n} \frac{\partial \mathcal{H}}{\partial x_i} \right)  \, dx_i$$

A entropia diferencial:

$$h(f) = - \int_S f(x) \log(f(x)) \, dx$$

A distribuição de máxima entropia de restrições perceptual-articulatória:

$$f_{\text{MaxEnt}}(F1,F2) = e^{- 1 - \lambda_0 - \lambda_1 \cdot v^c_P(F1) - \lambda_2 \cdot v^c_A(F1,F2)}$$

E a divergência de Kullback-Leibler:

$$\mathbf{KL}(p||q) = \int_{-\infty}^{\infty} p \log\left( \frac{p}{q} \right) \, dx$$

## Licença

Licença Pública Geral GNU (GPLv3) -- liberdade de compartilhar e alterar todas as versões desse script, na garantia que ele permaneça livre para todos os usuários.
