from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np

import streamlit as st
from PIL import Image
from skimage.transform import resize

# Configurações da página
st.set_page_config(page_title="DermAI", page_icon="imgs/Ico.png")

with st.sidebar:
    #logo = st.image("imgs/Logo-Sidebar.png")
    pagSelecionada = st.selectbox("Menu", ["Home", "DermAI", "About", "Contact"], help="Clique para selecionar a página")


if pagSelecionada == "Home":
    st.title("Seja bem-vindo!")
    st.markdown("<h3>DermAI, facilitando o seu diagnóstico!</h3>", unsafe_allow_html=True)

elif pagSelecionada == "DermAI":
    st.title("DermAI")

    imgUploader = st.file_uploader("Envie sua imagem:", type=[".png", ".jpg", ".jpeg", ".gif"], accept_multiple_files=False)

    if imgUploader is not None:
        st.success("Imagem recebida!")
        st.image(imgUploader, width=200)


        ## Parte da implementação do modelo e Funcionalidade

        # # Caminho do modelo utilizado
        # MODEL_PATH = 'models/modelo_aqui.h5' # ou .keras

        # width_shape = 224
        # height_shape = 224

        # # Classes do modelo
        # names = ['classe 1', 'classe 2']

        # # Receber a imagem no modelo e retornar a predição

        # def model_prediction(img, model):

        #     img_resize = resize(img, (width_shape, height_shape))
        #     x = preprocess_input(img_resize * 255)
        #     x = np.expand_dims(x, axis=0)

        #     pred = model.predict(x)
        #     return pred

        # def main():

        #     model = ''

        #     if model == '':
        #         model = load_model(MODEL_PATH)

        #     st.title("DermAI - Identificador de Dermatoses")

        #     predictS = ""
        #     img_file_buffer = st.file_uploader("Envie sua imagem:", type=[".png", ".jpg", ".jpeg", ".gif"], accept_multiple_files=False)

        #     if img_file_buffer is not None:
        #         image = np.array(Image.open(img_file_buffer))
        #         st.image(image, caption="Imagem", use_column_width=False)

        #     if st.button("Predição"):
        #         predictS = model_prediction(image, model)
        #         st.success('O diagnóstico é: {}'.format(names[np.argmax(predictS)]))

        # if __name__ == '__main__':
        #     main()

        pass

elif pagSelecionada == "About":

    st.title("O que é o DermAI?")

    # Aplicar CSS no Streamlit - tamanho da fonte
    st.markdown("""
    <style>
        .big-font{
                font-size: 17px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # st.markdown("<h3>Contextualização</h3>", unsafe_allow_html=True)

    # Texto da Introdução - Cada linha remete a um parágrafo
    st.markdown("<p class='big-font'>Comumente, a dificuldade no diagnóstico precoce de dermatoses está associada à falta de acesso a profissionais especializados, à escassez de recursos, recorrendo à realização de biópsias e à demora nos encaminhamentos. Essa problemática é agravada pelo fato de que muitas vezes os sintomas podem ser confundidos com outras condições dermatológicas, retardando o início do tratamento adequado. Dermatose é um termo abrangente que engloba diversas enfermidades cutâneas, apresentando desafios significativos tanto para pacientes quanto para profissionais de saúde em seu diagnóstico e tratamento.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>Além disso, a falta de conscientização e educação sobre cuidados dermatológicos também contribui para agravar a situação. Muitas pessoas podem não reconhecer os sinais de uma dermatose em estágios iniciais, adiando assim a busca por tratamento adequado. A ausência de programas educacionais sobre saúde da pele e a disseminação de informações imprecisas ou desatualizadas podem resultar em subdiagnóstico e subtratamento das dermatoses.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>Dado o aumento estimado de 20% na incidência de câncer no Brasil, incluindo o de pele, segundo Santos (2023), espera-se que, até 2030, ocorram mais de 25 milhões de casos novos, aumentando consequentemente a incidência de afecções dermatológicas na população e a dificuldade de alcançar um diagnóstico e tratamento precoces.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>Portanto, é essencial abordar não apenas as limitações de acesso aos serviços de saúde, mas também a necessidade de educação e conscientização da população sobre a importância do autocuidado dermatológico e da busca por assistência médica especializada quando necessário.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>A implementação de estratégias eficazes para a educação em saúde dermatológica pode desempenhar um papel fundamental na prevenção e no manejo adequado das dermatoses, complementando assim a proposta deste trabalho. Nesse sentido, a proposta deste trabalho busca oferecer uma solução inovadora que possa auxiliar na superação desses desafios, proporcionando uma ferramenta eficiente e acessível para o reconhecimento e diagnóstico das dermatoses.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>De forma geral, o objetivo presente neste trabalho se propõe a desenvolver um aplicativo destinado a pessoas com afecções dermatológicas, especificamente dermatoses, com o intuito de proporcionar uma ferramenta eficaz para o reconhecimento e diagnóstico breve dessas condições. Desta forma, o aplicativo visa utilizar recursos de inteligência artificial para identificar dermatoses por meio do reconhecimento de imagens, proporcionando diagnósticos precisos e ágeis. Ao capturar uma imagem da área afetada, o algoritmo embarcado no aplicativo realiza análises instantâneas, baseadas em um extenso treinamento de inteligência artificial. Essa abordagem não apenas agiliza o processo diagnóstico, mas também a orientação dos usuários em relação à escolha de profissionais médicos especializados e sugestões de tratamentos seguros.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>Justifica-se a relevância deste estudo pela crescente incidência de problemas de pele na população, aliada à escassez de recursos e à falta de acesso a profissionais especializados. O desenvolvimento de um aplicativo como o proposto não apenas contribui para a melhoria da qualidade de vida dos pacientes, ao permitir um diagnóstico antecipado e preciso, mas também representa um avanço significativo na área da saúde digital, possibilitando uma abordagem inovadora e acessível para o tratamento das dermatoses.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>A estrutura do projeto segue com uma abordagem metodológica clara e organizada, compreendendo diferentes etapas essenciais para o desenvolvimento do aplicativo. Inicialmente, foi realizada uma revisão bibliográfica abrangente sobre dermatoses, destacando os desafios no diagnóstico e tratamento, bem como as lacunas existentes na abordagem atual. Em seguida, foram delineados os objetivos específicos do projeto, que incluem o desenvolvimento do algoritmo de inteligência artificial para reconhecimento de imagens, a criação da interface do aplicativo e a implementação de funcionalidades para orientação dos usuários em relação aos profissionais médicos e tratamentos adequados.</p>", unsafe_allow_html=True)

    st.markdown("<p class='big-font'>A metodologia inclui a coleta de dados, o treinamento do algoritmo e a avaliação da precisão diagnóstica do aplicativo. Além disso, foi elaborado um plano de comunicação e divulgação do aplicativo, visando sua popularização e adesão pelos usuários-alvo. Por fim, a análise dos resultados obtidos e as considerações finais sendo apresentadas, destacando as contribuições do projeto para a área da saúde digital e as perspectivas de desenvolvimento e aprimoramento da ferramenta.</p>", unsafe_allow_html=True)
    
elif pagSelecionada == "Contact":

    st.title("Equipe")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("imgs/luiz-tcc.jpg", width=200, clamp=True)
        st.write("Luiz Henrique Góes Rodrigues")
        st.write("lhgrodrigues@dermai.com")
     

    with col2:
        st.image("imgs/luana-tcc.jpg", width=200, clamp=True)
        st.write("Luana Brisola Mena")
        st.write("lbmena@dermai.com")


    with col3:
        st.image("imgs/lucas-tcc.jpg", width=200, clamp=True)
        st.write("Lucas Marinelli Maciel")
        st.write("lmmaciel@dermai.com")


