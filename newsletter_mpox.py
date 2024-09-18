import tabula
import os
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Layout da página
st.set_page_config(
    page_title="Monitoramento Nacional MPOX",
    layout="wide"
)


##########################  EXTRAÇÃO E TRATAMENTO DOS ARQUIVOS EM PDF - INFORMES DO MINISTÉRIO DA SAÚDE ########################## 
@st.cache_data
def extract_tables_from_pdfs(directory_path, page_number):
    dataframes = {}
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        tables = tabula.read_pdf(pdf_path, pages=page_number, multiple_tables=True)
        for i, table in enumerate(tables):
            table_name = f"{pdf_file.replace('.pdf', '')}_table_{i}"
            dataframes[table_name] = table
    return dataframes

directory_path = r'C:\Users\Gleice Barros\OneDrive - svriglobal.com\Área de Trabalho\MPOX\Informes COE MPOX'
page_number = 5
dataframes = extract_tables_from_pdfs(directory_path, page_number)

for name, df in dataframes.items():
    print(f"DataFrame Name: {name}")
    print(df.head())
    print("\n")

##########################  SELECIONANDO INFORMAÇÕES DE INTERESSE - INFORMES DO MINISTÉRIO DA SAÚDE ########################## 
def combine_rows_with_source(dataframes, rows_to_extract):
    combined_df = pd.DataFrame()  

    for name, df in dataframes.items():
        try:
            rows = df.iloc[[row - 1 for row in rows_to_extract]]
            rows['Source_File'] = name
            combined_df = pd.concat([combined_df, rows], ignore_index=True)
        except IndexError as e:
            print(f"Erro ao extrair linhas para {name}: {e}")

    return combined_df

rows_to_extract = [7, 9, 13, 21, 28, 30]  
combined_df = combine_rows_with_source(dataframes, rows_to_extract)

# Limpeza e formatação do DataFrame
def split_and_convert(value):
    parts = value.split(' ', 1)
    return (int(parts[0]), int(parts[1])) if len(parts) == 2 else (int(parts[0]), 0)

combined_df[['2023:Casos', '2023:Óbitos']] = combined_df['Ano de Notificação/Evolução'].apply(lambda x: pd.Series(split_and_convert(x)))
combined_df.drop(columns='Ano de Notificação/Evolução', inplace=True)

# Renomear colunas
combined_df.drop(columns=['Unnamed: 2', 'Unnamed: 5'], inplace=True)
new_column_names = {
    'Unnamed: 0': 'UF de Residência',
    'Unnamed: 1': '2022:Casos',
    'Unnamed: 3': '2022:Óbitos',
    'Unnamed: 4': '2024:Casos',
    'Unnamed: 6': '2024:Óbitos'
}
combined_df.rename(columns=new_column_names, inplace=True)

# Reordenar colunas
new_order = ['Source_File', 'UF de Residência', '2022:Casos', '2022:Óbitos', '2023:Casos', '2023:Óbitos', '2024:Casos', '2024:Óbitos']
combined_df = combined_df[new_order]

# Reorganização da coluna Source_File para conseguir identificar qual a semana e data da fonte dos Dados. 
def extrair_semana_data(texto):
    
    match_semana = re.search(r'Informe Semanal nº (\d+)', texto)
    semana_num = match_semana.group(1) if match_semana else 'Desconhecida'
    
    match_data = re.search(r'(\d{2}) de ([A-Za-z]+) de (\d{4})', texto)
    if match_data:
        dia = match_data.group(1)
        mes = match_data.group(2)
        ano = match_data.group(3)
        
        meses = {
            'Janeiro': 'Jan', 'Fevereiro': 'Fev', 'Março': 'Mar', 'Abril': 'Abr',
            'Maio': 'Mai', 'Junho': 'Jun', 'Julho': 'Jul', 'Agosto': 'Ago',
            'Setembro': 'Set', 'Outubro': 'Out', 'Novembro': 'Nov', 'Dezembro': 'Dez'
        }
        mes_abreviado = meses.get(mes, mes[:3]) 
        
        return f'Semana {semana_num} - {dia}.{mes_abreviado}'
    return f'Semana {semana_num} - Data Desconhecida'

# Aplicar a função no DataFrame
combined_df['Source_File'] = combined_df['Source_File'].apply(extrair_semana_data)

#Substituição de Nan por zero (0)
combined_df.fillna(0, inplace=True)

#Função para limpeza nos números dos casos nas colunas dos anos.
combined_df = combined_df.map(lambda x: x.strip() if isinstance(x, str) else x)
def format_number(x):
    if isinstance(x, str): 
        return x.replace('.', '') 
        x = int(x)
    return x

# Aplicando a função nas colunas 
combined_df['2022:Casos'] = combined_df['2022:Casos'].apply(format_number)
combined_df['2022:Óbitos'] = combined_df['2022:Óbitos'].apply(format_number)
combined_df['2023:Casos'] = combined_df['2023:Casos'].apply(format_number)
combined_df['2023:Óbitos'] = combined_df['2023:Óbitos'].apply(format_number)
combined_df['2024:Casos'] = combined_df['2024:Casos'].apply(format_number)
combined_df['2024:Óbitos'] = combined_df['2024:Óbitos'].apply(format_number)


#### Inserção dos dados da primeira semana de informe, devido a não ter tabela destacada no arquivo PDF desta Semana #####
novas_linhas = pd.DataFrame({
    'Source_File': ['Semana 01 - 20.Ago'] * 6,
    'UF de Residência': ['BA', 'DF', 'MG', 'RJ', 'SP', 'Total'],
    '2022:Casos': [164, 312, 636, 1393, 4153, 10648],
    '2022:Óbitos': [0, 0, 3, 5, 3, 14],
    '2023:Casos': [37, 26, 60, 173, 155, 853],
    '2023:Óbitos': [0, 0, 1, 0, 0, 2],
    '2024:Casos': [34, 17, 50, 187, 400, 791],
    '2024:Óbitos': [0, 0, 0, 0, 0, 0]
})

combined_df = pd.concat([novas_linhas, combined_df], ignore_index=True)

combined_df = combined_df.astype({'2022:Casos': 'int16'})
combined_df = combined_df.astype({'2024:Casos': 'int16'})
combined_df['Source_File'] = combined_df['Source_File'].astype(str)
########################## GERANDO MÉTRICAS PARA DEMONSTRAÇÃO E ATUALIZAÇÃO SEMANAL ##########################


caminho_logo = 'Logo svri texto preto.png'
st.sidebar.image(caminho_logo, use_column_width=True)

st.write("# Monitoramento de casos MPOX - Brasil")

multi = '''Este informativo foi elaborado com o objetivo de acompanhamento analítico dos números confirmados de contágio por MPOX no Brasil. 
Os dados são obtidos por meio dos Informes Semanais publicados pelo Ministério da Saúde (MS), através do Centro de Operações de Emergência para MPOX (COE/MPOX). O primeiro informe foi publicado em 20 de agosto de 2024, após decretação da Emergência de Saúde Pública de Importância Internacional (ESPII) pela Organização Mundial da Saúde (OMS) em 14 de agosto de 2024.

Até o momento, não foram registrados óbitos e os casos confirmados no Brasil são do Clado 2B da doença, diferente do novo Clado (1b), responsável por maior parte dos casos recentes na região central do Continente Africano.

Vale ressaltar ainda: os dados da Semana 01 - 20.ago são os números acumulados desde o inicio do ano de 2024, computados pelo MS.
'''
st.sidebar.markdown(multi)


col1, col2, col3, col4 = st.columns(4)
with col4:
    def get_last_update_time():
        return datetime.now()

    def format_datetime(dt):
        return dt.strftime('%d/%m/%Y %H:%M:%S')
    
    last_update_time = get_last_update_time()
    st.write(f'Última atualização: {format_datetime(last_update_time)}')

with col1:
    def obter_total_casos_semana_mais_recente(df):
   
        semanas = df['Source_File'].unique()
        semanas = sorted(semanas, key=lambda x: re.sub(r'\D', '', x)) 
        semana_mais_recente = semanas[-1]

        df_semana_recente = df[df['Source_File'] == semana_mais_recente]

        total_linha = df_semana_recente[df_semana_recente['UF de Residência'] == 'Total']
        if not total_linha.empty:
            total_casos = total_linha['2024:Casos'].values[0]
            return semana_mais_recente, total_casos
        return semana_mais_recente, 0

    semana_recente, total_casos = obter_total_casos_semana_mais_recente(combined_df)

    st.metric(label=f'TOTAL ATUALIZADO - {semana_recente}', value=f'{total_casos}')

with col2:
    df_total = combined_df[combined_df['UF de Residência'] == 'Total']
    df_total['Aumento (%)'] = df_total['2024:Casos'].pct_change() * 100
    aumento_ultima_semana = df_total['Aumento (%)'].iloc[-1] if not df_total.empty else 0
    st.metric(
        label="Aumento(%) em relação a semana anterior",
        value=f"{aumento_ultima_semana:.2f}%",
        delta=f"{aumento_ultima_semana:.2f}%"
    )
df_total['2022:Óbitos'] = pd.to_numeric(df_total['2022:Óbitos'], errors='coerce')
df_total['2022:Óbitos'].fillna(0, inplace=True)
df_total['Semana_Num'] = df_total['Source_File'].str.extract('(\d+)').astype(int)

with col3:
    df_total = combined_df[combined_df['UF de Residência'] == 'Total']
    if not df_total.empty:
        valor_primeira_semana = df_total['2024:Casos'].iloc[0]
        valor_ultima_semana = df_total['2024:Casos'].iloc[-1] 
        aumento_percentual = ((valor_ultima_semana - valor_primeira_semana) / valor_primeira_semana) * 100
    else:
        aumento_percentual = 0

    st.metric(
        label="Aumento (%) em relação à primeira semana",
        value=f"{aumento_percentual:.2f}%",
        delta=f"{aumento_percentual:.2f}%"
    )


########################## GERANDO OS GRAFICOS COM AS INFORMAÇÕES COLHIDAS ATÉ AQUI ##########################
col4, col5 = st.columns(2)

#Grafico dos Dados por Estado
with col5:
    df_filtrado = combined_df.loc[combined_df['UF de Residência'] != 'Total']

    color_sequence = ['#EC0E73', '#041266', '#00A1E0', '#C830A0', '#61279E']

    fig = go.Figure()

    for uf in df_filtrado['UF de Residência'].unique():
        df_subset = df_filtrado[df_filtrado['UF de Residência'] == uf]
        fig.add_trace(go.Bar(
            x=df_subset['Source_File'],
            y=df_subset['2024:Casos'],
            name=uf,
            marker_color=color_sequence[df_filtrado['UF de Residência'].unique().tolist().index(uf)],
            text=df_subset['2024:Casos'],
            texttemplate='%{text}', 
            textposition='outside' 
        ))

    fig.update_layout(
        title= 'MPOX: 5 Estados x Casos Confirmados',
        xaxis_title='',
        yaxis_title='Número de Casos',
        barmode='group',
        yaxis=dict(range=[0, 700])
    )

    st.plotly_chart(fig)

# Grafico com os dados anuais de MPOX
with col4:
    df_total = combined_df.loc[combined_df['UF de Residência'] == 'Total']
    ultima_semana = df_total.iloc[-1]
    df_ultima_semana = pd.DataFrame([ultima_semana])

    fig1 = go.Figure(data=[
        go.Bar(name='2022', x=df_ultima_semana['Source_File'], y=df_ultima_semana['2022:Casos'], text=df_ultima_semana['2022:Casos'], textposition='auto', marker_color='#EC0E73'),  # Magenta
        go.Bar(name='2023', x=df_ultima_semana['Source_File'], y=df_ultima_semana['2023:Casos'], text=df_ultima_semana['2023:Casos'], textposition='auto', marker_color='#041266'),  # Azul Escuro
        go.Bar(name='2024', x=df_ultima_semana['Source_File'], y=df_ultima_semana['2024:Casos'], text=df_ultima_semana['2024:Casos'], textposition='auto', marker_color='#00A1E0')   # Azul Claro
    ])

    fig1.update_layout(
        title='Evolução Casos MPOX no Brasil 2022-24',
        xaxis_title='Valores Totais por Ano',
        yaxis_title='Número de Casos',
        barmode='group',
        legend_title='ANO',
    )

    fig1.update_yaxes(range=[0, 11000])
    st.plotly_chart(fig1)

# Gráfico com a evolução dos numeros de casos semana a semana

df_total['Semana_Num'] = df_total['Source_File'].str.extract(r'(\d+)').astype(int)

fig2 = px.line(df_total, x='Source_File', y='2024:Casos', text='2024:Casos')
fig2.update_traces(
    line=dict(color='#EC0E73', width=3), 
    textposition='top center',  
    texttemplate='%{text:.0f}',
    name='Número de Casos Reais',
    showlegend=True 
)

fig2.update_layout(
    title='MPOX - 2024: Número de Casos por Semana',
    xaxis_title='Informes COE/MPOX',
    yaxis_title='Número de Casos',
    xaxis=dict(showgrid=True, zeroline=False),  
    yaxis=dict(showgrid=True, zeroline=False),  
    plot_bgcolor='rgba(245, 245, 245, 0.85)',
    showlegend=True 
)

# Regressão Linear
X = df_total['Semana_Num'].values.reshape(-1, 1) 
y = df_total['2024:Casos'].values

modelo = LinearRegression()
modelo.fit(X, y)

semanas_futuras = pd.DataFrame({'Semana_Num': [6, 7, 8]}) 
predicoes_futuras = modelo.predict(semanas_futuras)

df_futuro = pd.DataFrame({
    'Source_File': [f'Semana {i} - Data Futuro' for i in semanas_futuras['Semana_Num']],
    '2024:Casos': predicoes_futuras 
})

df_total = pd.concat([df_total, df_futuro], ignore_index=True)

fig2.add_scatter(
    x=df_total['Source_File'], 
    y=df_total['2024:Casos'], 
    mode='lines+markers+text', 
    name='Previsão de Número de Casos', 
    line=dict(color='blue', dash='dash'),
    text=df_total['2024:Casos'], 
    textposition='top center')

st.plotly_chart(fig2)

st.dataframe(combined_df)

st.markdown("Fonte: Informes MPOX - https://www.gov.br/saude/pt-br/composicao/svsa/coes/mpox/informes")

st.sidebar.markdown("Desenvolvido por [Science Valley Research Institute](https://svriglobal.com/)")
