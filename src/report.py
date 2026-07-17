import locale
import os
import time
import subprocess
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer

def generate(cenarios):
    pdf = SimpleDocTemplate("outputs/ChurnIA-ProjecoesML.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    logo = Image("img/logo_sctecr.png", width=178, height=47)
    story.append(logo)      
    #story.append(Spacer(1, 20))
    locale.setlocale(locale.LC_TIME, "pt_BR.UTF-8")
    agora = datetime.now()
    story.append(Paragraph("Projeto: <b>ChurnIA</b>", styles["Normal"]))
    story.append(Paragraph("Geração: <b>" + agora.strftime("%A, %d de %B de %Y %H:%M") + "</b>", styles["Normal"]))
    story.append(Paragraph("Setores: <b>Contábil, TI e Staff</b>", styles["Normal"]))
    story.append(Paragraph("Projeção: <b>07/2026 – 12/2026</b>", styles["Normal"]))
    story.append(Paragraph("Responsável: <b>Julio Cesar Bueno de Oliveira</b>", styles["Normal"]))
    story.append(Paragraph("Cliente: <b>SCTec - Análise Preditiva com Python [T2]</b>", styles["Normal"]))    
    story.append(Spacer(1, 10))

    texto = """Este relatório tem como propósito apresentar os resultados das projeções
                realizadas a partir dos dados utilizados no treinamento do modelo de Machine Learning.
                O objetivo é fornecer uma visão estruturada e confiável sobre os cenários analisados,
                destacando os principais riscos e tendências identificados."""
    story.append(Paragraph(texto, styles["Normal"]))
    story.append(Spacer(1, 10))

    for titulo, df in cenarios.items():
        story.extend(df_tabela(df, f"Previsão - {titulo}"))

    styles = getSampleStyleSheet()    

    story.append(Spacer(1, 10))
    story.append(Paragraph("Projeções de Risco de Churn", styles["Heading2"]))
    #story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Cenário Base (Normal)</b>", styles["Heading3"]))
    story.append(Paragraph("Todos os meses, menos Agosto e Setembro, permanecem em <b>BAIXO RISCO</b>.", styles["Normal"]))
    story.append(Paragraph("Indica estabilidade e baixa probabilidade de churn prolongado nas condições atuais.", styles["Normal"]))
    #story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Cenário 1 (10% aumento de receita e custos)</b>", styles["Heading3"]))
    story.append(Paragraph("Comportamento onde todos os meses estão em <b>BAIXO RISCO</b>.", styles["Normal"]))
    story.append(Paragraph("Pequenos ajustes proporcionais ajudam significativamente a eliminar o risco.", styles["Normal"]))
    #story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Cenário 2 (20% aumento apenas de custos)</b>", styles["Heading3"]))
    story.append(Paragraph("Julho, Agosto, Setembro e Outubro passam para <b>ALTO RISCO</b>.", styles["Normal"]))
    story.append(Paragraph("Demonstra que aumento isolado de custos compromete a sustentabilidade, epode levar a um churn prolongado.", styles["Normal"]))
    #story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Cenário 3 (50% aumento de despesas)</b>", styles["Heading3"]))
    story.append(Paragraph("Julho, Agosto, Setembro e Outubro mantêm classificação de <b>ALTO RISCO</b>.", styles["Normal"]))
    story.append(Paragraph("Novembro e Dezembro retornam para <b>BAIXO RISCO</b>, sugerindo recuperação parcial.", styles["Normal"]))
    story.append(Paragraph("Evidencia vulnerabilidade significativa diante de pressões financeiras nestes meses, é bom ficar de olho.", styles["Normal"]))
    #story.append(Spacer(1, 20))

    story.append(Paragraph("Conclusão", styles["Heading2"]))    
    story.append(Paragraph("O risco cresce de forma sensível quando há <b>pressão de custos sem contrapartida de receita</b>.", styles["Normal"]))
    story.append(Paragraph("Meses críticos: <b>Agosto a Outubro</b>, que exigem atenção especial.", styles["Normal"]))
    story.append(Paragraph("Recomenda-se implementar <b>ações preventivas de retenção</b> e <b>controle de despesas</b> nesses períodos para mitigar riscos.", styles["Normal"]))
    pdf.build(story)

    print("==================================================================")
    print("Relatório de projeção gerado com sucesso!")
    print("Favor verificar no local abaixo.")
    print("outputs/ChurnIA-ProjecoesML.pdf")
    print("==================================================================")


def df_tabela(df, titulo):
    data = [df.columns.tolist()] + df.values.tolist()
    tabela = Table(data)
    tabela.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),1,colors.black),
    ]))
    return [Paragraph(f"<b>{titulo}</b>", getSampleStyleSheet()["Heading3"]), Spacer(1,5), tabela, Spacer(1,10)]

def open(nome_arquivo: str = "ChurnIA-ProjecoesML.pdf"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho = os.path.join(base_dir, "outputs", nome_arquivo)

    if os.path.exists(caminho):
        try:
            if os.name == 'nt':  # Windows
                os.startfile(caminho)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(["xdg-open", caminho])
            else:
                print("Sistema operacional não suportado.")
        except Exception as e:
            print(f"Erro ao abrir o PDF: {e}")
    else:
        print("Arquivo não encontrado:", caminho)
