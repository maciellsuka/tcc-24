import streamlit as st
from PIL import Image

import argparse
import logging
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

# Configurações da página
st.set_page_config(page_title="DermAI", page_icon="imgs/Ico.png")

with st.sidebar:
    logo = st.image("imgs/Logo-Sidebar.png")
    pagSelecionada = st.selectbox("Menu", ["Index", "DermAI", "About", "Contact"])

if pagSelecionada == "Index":
    st.title("Seja bem-vindo!")
    st.markdown("<h3>DermAI, facilitando o seu diagnóstico!</h3>", unsafe_allow_html=True)

elif pagSelecionada == "DermAI":
    st.title("DermAI")

    imgUploader = st.file_uploader("Envie sua imagem:", type=[".png", ".jpg", ".jpeg", ".gif"], accept_multiple_files=False)

    if imgUploader is not None:

        if os.path.exists('output'):
            shutil.rmtree('output')

        if not os.path.exists('output'):
            os.makedirs('output')
        # Carregar e exibir a imagem
        image = Image.open(imgUploader)
        st.image(image, caption='Imagem enviada', use_column_width=True)
        
        # Salva a imagem na pasta 'outputs'
        image_path = os.path.join('output', imgUploader.name)
        image.save(image_path)
        
        st.success(f"Imagem salva em: {image_path}")

        def get_segment_crop(img, mask, cl=[0]):
            img[~np.isin(mask, cl)] = 0
            return img

        def predict_img(net,
                        full_img,
                        device,
                        image_size=(256, 256),
                        out_threshold=0.5,
                        out_mask_filename='mask.png'):
            net.eval()
            img = torch.from_numpy(BasicDataset.preprocess(None, full_img, image_size, is_mask=False))
            img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                output = net(img).cpu()
                print(f'Ouput shape: {output.shape}')
                output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
                
                if net.n_classes > 1:
                    mask = output.argmax(dim=1)

                    # Save all crops masks
                    mask_filename = out_mask_filename[:out_mask_filename.rfind('.')]
                    for cl, mask_class in enumerate(output[0]):
                        mask_reshaped = mask.numpy().reshape((mask.shape[1], mask.shape[2]))
                        # Crop each class
                        full_img_cropped = get_segment_crop(np.array(full_img), mask=mask_reshaped, cl=[cl])
                        full_img_cropped = Image.fromarray(full_img_cropped)
                        full_img_cropped.save(f'{mask_filename}-{cl}.png')

                        '''
                        mask_class = torch.sigmoid(mask_class) > out_threshold

                        # Mask without argmax (only using threshold)
                        mask_class_without_argmax = Image.fromarray(mask_class.numpy().astype(bool))
                        mask_class_without_argmax.save(f'{mask_filename}-{cl}-mask_without_argmax.png')

                        # Mask using argmax
                        only_mask_class = get_segment_crop(mask_class.numpy().astype(bool), mask=mask_reshaped, cl=[cl])
                        only_mask_class = Image.fromarray(only_mask_class)
                        only_mask_class.save(f'{mask_filename}-{cl}-mask.png')
                        '''

                    full_img_cropped = get_segment_crop(np.array(full_img), mask=mask_reshaped, cl=range(1, output.shape[1]))
                    full_img_cropped = Image.fromarray(full_img_cropped)
                    full_img_cropped.save(f'{mask_filename}-all_class.png')

                else:
                    mask = torch.sigmoid(output) > out_threshold

            return mask[0].long().squeeze().numpy()


        def get_args():
            parser = argparse.ArgumentParser(description='Predict masks from input images')
            parser.add_argument('--model', '-m', default='model/checkpoint_epoch11.pth', metavar='FILE',
                                help='Specify the file in which the model is stored')
            parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default=['output'],
                                help='Filenames or folder of input images')
            parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
            parser.add_argument('--viz', '-v', action='store_true',
                                help='Visualize the images as they are processed')
            parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
            parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                                help='Minimum probability value to consider a mask pixel white')
            parser.add_argument('--image_size', '-s', type=tuple, default=(256, 256),
                                help='Resize images')
            parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
            parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
            
            return parser.parse_args()


        def get_output_filenames(args):
            def _generate_name(fn):
                return f'{os.path.splitext(fn)[0]}_OUT.png'

            return args.output or list(map(_generate_name, args.input))


        def mask_to_image(mask: np.ndarray, mask_values):
            if isinstance(mask_values[0], list):
                out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
            elif mask_values == [0, 1]:
                out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
            else:
                out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
                # Add a color for each class (grayscale)
                interval_colors = 255 / (len(mask_values) - 1)
                for idx, v in enumerate(mask_values):
                    mask_values[idx] = int(idx * interval_colors)

            if mask.ndim == 3:
                mask = np.argmax(mask, axis=0)

            for i, v in enumerate(mask_values):
                out[mask == i] = v

            return Image.fromarray(out)


        if __name__ == '__main__':
            args = get_args()
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

            if os.path.isdir(args.input[0]):
                filenames = os.listdir(args.input[0])
                in_files = []
                for filename in filenames:
                    in_files.append(f'{args.input[0]}/{filename}')
                args.input = in_files

            in_files = args.input

            out_files = get_output_filenames(args)

            net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f'Loading model {args.model}')
            logging.info(f'Using device {device}')

            net.to(device=device)
            state_dict = torch.load(args.model, map_location=device)
            mask_values = state_dict.pop('mask_values', [0, 1])
            net.load_state_dict(state_dict)

            logging.info('Model loaded!')

            for i, filename in enumerate(in_files):
                logging.info(f'Predicting image {filename} ...')
                img = Image.open(filename)

                mask = predict_img(net=net,
                                full_img=img,
                                image_size=args.image_size,
                                out_threshold=args.mask_threshold,
                                device=device,
                                out_mask_filename=out_files[i])

                if not args.no_save:
                    out_filename = out_files[i]
                    result = mask_to_image(mask, mask_values)
                    result.save(out_filename)
                    logging.info(f'Mask saved to {out_filename}')

                if args.viz:
                    logging.info(f'Visualizing results for image {filename}, close to continue...')
                    plot_img_and_mask(img, mask)
        
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
