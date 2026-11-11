"""Create 100 pages: two substantive sections for each of 50 fictional policies."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from PIL import Image as RasterImage, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF = ROOT / 'output/pdf/policy_handbook.pdf'

def main():
    PDF.parent.mkdir(parents=True, exist_ok=True)
    topics = json.loads((ROOT / 'policy_topics.json').read_text())
    assert len(topics) == 50, 'Use 50 topics to produce exactly 100 pages.'
    # A raster workflow figure, with the same information in a selectable caption.
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    diagram = assets / 'request_workflow.png'
    picture = RasterImage.new('RGB', (1470, 150), 'white')
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 27) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
    for index, label in enumerate(('Record request', 'Check conditions', 'Decide or escalate')):
        x = index * 500
        draw.rounded_rectangle((x + 5, 15, x + 455, 130), radius=15, fill='#dcebf0', outline='#15374b', width=3)
        draw.text((x + 230, 73), label, fill='#15374b', font=font, anchor='mm')
        if index < 2:
            draw.line((x + 460, 73, x + 493, 73), fill='#15374b', width=4)
            draw.polygon([(x + 493, 73), (x + 480, 63), (x + 480, 83)], fill='#15374b')
    picture.save(diagram)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PolicyTitle', fontSize=21, leading=26, spaceAfter=18, textColor=colors.HexColor('#15374b')))
    styles['BodyText'].leading = 15
    styles['BodyText'].spaceAfter = 10
    def para(text, style='BodyText'):
        return Paragraph(escape(text), styles[style])
    story = []
    for i, (title, rule, procedure, example, question) in enumerate(topics, 1):
        for part in ('Policy and eligibility', 'Procedure and case handling'):
            page = len([x for x in story if isinstance(x, PageBreak)]) + 1
            story += [para(f'POLICY P{i:03d} | {part}', 'Heading2'), para(title, 'PolicyTitle')]
            story += [para('Scope and rule', 'Heading2'), para(rule)]
            if part == 'Policy and eligibility':
                story += [para('Terms and conditions', 'Heading2')]
                for term in [rule, 'Apply this rule to the stated situation only; do not infer approval for other cases.',
                             'Keep the original purchase or service record so the stated dates and conditions can be verified.']:
                    story.append(Paragraph('&#8226; ' + escape(term), styles['BodyText']))
                rows = [('Field', 'Policy requirement'), ('Policy code', f'P{i:03d}'), ('Rule', rule), ('Worked example', example)]
            else:
                story += [para('Required procedure', 'Heading2'), para(procedure), para('Conditions for handling', 'Heading2')]
                for term in ['Record the policy code, request date, and customer-provided evidence.',
                             'Check all explicit deadlines and exceptions before deciding the outcome.',
                             'If the policy does not specify a requested detail, escalate rather than invent a rule.']:
                    story.append(Paragraph('&#8226; ' + escape(term), styles['BodyText']))
                rows = [('Stage', 'Action or evidence'), ('Assessment', procedure), ('Example', example), ('Practice question', question)]
            story += [Spacer(1, 8), para('Reference table', 'Heading2')]
            table = Table([[para(a), para(b)] for a,b in rows], colWidths=[105, 385])
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),
                ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#9bb2bd')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story += [table]
            if part == 'Procedure and case handling' and i % 5 == 0:
                story += [Spacer(1, 10), Image(str(diagram), width=490, height=50),
                          para(f'Figure {i // 5}: {title} workflow. Record the request, check the policy conditions, then decide or escalate. The diagram is a summary; the rules above remain controlling.')]
            if page < 100:
                story.append(PageBreak())
    def furniture(canvas, doc):
        canvas.setFillColor(colors.HexColor('#15374b'))
        canvas.setFont('Helvetica', 9)
        canvas.drawString(50, 815, 'NORTHSTAR | ADVANCED RAG POLICY HANDBOOK')
        canvas.drawString(50, 28, 'FICTIONAL TRAINING DATA | NOT A REAL BUSINESS POLICY')
        canvas.drawRightString(545, 28, f'{doc.page} / 100')
    SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=50, rightMargin=50,
                      topMargin=65, bottomMargin=50).build(story, onFirstPage=furniture, onLaterPages=furniture)
    pages = PdfReader(PDF).pages
    assert len(pages) == 100, f'Expected 100 pages; got {len(pages)}'
    assert all(len(p.extract_text().split()) > 100 for p in pages)
    assert sum(bool(p.images) for p in pages) == 10, 'Expected 10 illustrated pages.'
    print(f'Created {len(pages)} pages: {PDF}')

if __name__ == '__main__':
    main()
