
# GradiEnt (versão para qualificação)

Script para análise fonológica de variáveis fonéticas contínuas. Desenvolvido em 2024 durante minha pesquisa de pós-graduação. A versão apresentada aqui foi concluída enter março e abril de 2024, com intenção de apresentá-la para qualificação do Mestrado.

# Arquivos
Este repositório contém 3 arquivos de importância:

### GradiEnt.py:

Script para rodar as análises. Basta rodar no terminal com as amostras na mesma pasta, e.g.:

> $ python3 GradiEnt.py


### amostras.txt:

Arquivo que contém os valores dos formantes (F1 e F2) analisados. Para entender melhor sua formatação, basta ler este arquivo e comparar com a função *ler_dados()* do GradiEnt, que filtra elementos pelo cabeçalho contendo palavras-chave.

### GradiEnt_comentado.py:

Versão mais completa do GradiEnt que gera uma grande quantidade de *plots* (alguns deles para *debugging* e muitos deles inúteis para uma análise, só servindo para apoio visual durante minha escrita da qualificação) e está amplamente comentado. Logo, é consideravelmente mais didático se sua intenção for compreender o que está embaixo do capô.

# Equações

As principais equações que esse script roda são a restrição perceptual:

$$R_{P}(F1) = L - \frac{L^2}{(1 + e^{k_1(F1 - F1_{\text{limiar}_{1}})})(1 + e^{-k_2(F1 - F1_{\text{limiar}_{2}})})$$

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

Licença Pública Geral GNU (GPLv3) -- liberdade de compartilhar e alterar todas as versões, na garantia que ele permaneça livre para todos os usuários.
