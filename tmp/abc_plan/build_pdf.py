from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'instructions/KEHITYSSUUNNITELMA_ABC.md'
OUTPUT = ROOT / 'output/pdf/Sijoitusanalyysisovellus_kehityssuunnitelma_ABC.pdf'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-Bold')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyFI', fontName='Arial', fontSize=10.2, leading=14.1, spaceAfter=7, textColor=colors.black, splitLongWords=True))
styles.add(ParagraphStyle(name='TitleFI', fontName='Arial-Bold', fontSize=22, leading=26, spaceAfter=15, textColor=colors.black))
styles.add(ParagraphStyle(name='H1FI', fontName='Arial-Bold', fontSize=17, leading=21, spaceAfter=13, textColor=colors.black, keepWithNext=True))
styles.add(ParagraphStyle(name='H2FI', fontName='Arial-Bold', fontSize=11.8, leading=15, spaceBefore=9, spaceAfter=7, textColor=colors.black, keepWithNext=True))
styles.add(ParagraphStyle(name='ListFI', parent=styles['BodyFI'], leftIndent=12, firstLineIndent=-12, spaceAfter=5))
styles.add(ParagraphStyle(name='CellFI', fontName='Arial', fontSize=9, leading=12, spaceAfter=0, textColor=colors.black))
styles.add(ParagraphStyle(name='HeaderFI', parent=styles['CellFI'], fontName='Arial-Bold', textColor=colors.white))
styles.add(ParagraphStyle(name='SourceFI', parent=styles['BodyFI'], fontSize=8.5, leading=11.5, spaceAfter=7))

def inline(s):
    s = escape(s)
    s = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(https://[^\s<]+)', lambda m: '<link href="' + m.group(1) + '" color="#174A71">' + m.group(1) + '</link>', s)
    return s

def paragraph(text, style='BodyFI'):
    return Paragraph(inline(text), styles[style])

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._page_states = []
    def showPage(self):
        self._page_states.append(dict(self.__dict__))
        self._startPage()
    def save(self):
        page_count = len(self._page_states)
        for state in self._page_states:
            self.__dict__.update(state)
            self.setFont('Arial', 8)
            self.setFillColor(colors.HexColor('#555555'))
            self.drawString(48, 27, 'Kehityssuunnitelma  |  15.11.2026')
            self.drawRightString(A4[0]-48, 27, f'{self._pageNumber} / {page_count}')
            super().showPage()
        super().save()

story = []
blocks = SOURCE.read_text(encoding='utf-8').strip().split('\n\n')
for idx, block in enumerate(blocks):
    if block == '<!-- pagebreak -->':
        story.append(PageBreak())
    elif block.startswith('# '):
        story.append(paragraph(block[2:], 'TitleFI' if idx == 0 else 'H1FI'))
    elif block.startswith('## '):
        story.append(paragraph(block[3:], 'H2FI'))
    elif block.startswith('| '):
        rows = [line.strip().strip('|').split('|') for line in block.splitlines()]
        rows = [[s.strip() for s in row] for row in rows if not all(set(s.strip()) <= set('-: ') for s in row)]
        width = A4[0]-96
        cols = len(rows[0])
        if cols == 3:
            widths = [width*0.24, width*0.49, width*0.27]
            if 'Nykyinen osa' in rows[0][0]: widths = [width*0.25, width*0.35, width*0.40]
        else:
            widths = [width*0.35, width*0.65]
        table = Table([[paragraph(s, 'HeaderFI' if ri == 0 else 'CellFI') for s in row] for ri,row in enumerate(rows)], colWidths=widths, repeatRows=1, hAlign='LEFT')
        commands = [('BACKGROUND',(0,0),(-1,0),colors.HexColor('#203B53')), ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#D9D9D9')), ('VALIGN',(0,0),(-1,-1),'MIDDLE'), ('LEFTPADDING',(0,0),(-1,-1),8), ('RIGHTPADDING',(0,0),(-1,-1),8), ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7)]
        for row in range(1,len(rows)):
            commands.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#F0F4F7') if row%2 else colors.white))
        table.setStyle(TableStyle(commands))
        story.extend([Spacer(1,3),table,Spacer(1,10)])
    elif re.match(r'^(\d+\. |\- )', block):
        for line in block.splitlines():
            story.append(paragraph(line, 'ListFI'))
        story.append(Spacer(1,3))
    else:
        story.append(paragraph(block.replace('\n',' '), 'SourceFI' if re.match(r'^\[\d+\]', block) else 'BodyFI'))

doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=43, bottomMargin=47, title='Sijoitusanalyysin oppimisympäristön kehityssuunnitelma', author='Jari', subject='Microsoftin laskelmat ja USA-yhtiöiden oppimispolku sekä myöhemmät A ja B')
doc.build(story, canvasmaker=NumberedCanvas)
print(OUTPUT)

from pypdf import PdfReader
reader = PdfReader(OUTPUT)
print('Pages:', len(reader.pages))
for n,page in enumerate(reader.pages,1):
    text=page.extract_text()
    print(n, len(text), text.splitlines()[0])

import pypdfium2 as pdfium
pdf = pdfium.PdfDocument(str(OUTPUT))
render_dir = ROOT / 'tmp/abc_plan/render'
render_dir.mkdir(parents=True, exist_ok=True)
for n in range(len(pdf)):
    img = pdf[n].render(scale=1.5).to_pil()
    img.save(render_dir / f'page-{n+1}.png')
print('Rendered', len(pdf), 'pages')
