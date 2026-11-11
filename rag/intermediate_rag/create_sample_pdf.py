"""Rebuild the fictional 50-page document (requires requirements-build.txt).

Run: python rag/intermediate_rag/create_sample_pdf.py
This replaces sample_document.pdf in this folder. Rebuild embeddings afterward.
Every page has a different topic, a rule, a procedure, and a worked example.
"""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from pypdf import PdfReader


# title, rule, procedure, worked example, practice question
TOPICS = [
    ("Refund eligibility", "Customers may request a refund within 30 calendar days of purchase. The item must be unused, and a receipt is required.",
     "Ask for the order number, purchase date, and item condition. Compare the request date with the purchase date before approving a return. Used items do not qualify for a change-of-mind refund.",
     "An unused bag purchased 12 days ago with a receipt qualifies for a refund request. A used bag does not qualify under this rule.", "Can I return an unused item after 12 days?"),
    ("Starting a return", "A return authorization must be issued before a customer sends an item back. Authorization remains valid for 14 calendar days.",
     "Submit the order number and return reason through the account return form. Support supplies a return reference and packing instructions. Write the reference on the parcel label, not on the product packaging.",
     "A customer receives authorization on the first day of the month. The parcel must be handed to the carrier within the next 14 calendar days.", "How long is a return authorization valid?"),
    ("Return packaging", "Returned products must include their original accessories, manuals, and protective packaging. A missing accessory delays assessment until support reviews the case.",
     "Place the product box inside a separate shipping carton. Remove old shipping labels, cushion fragile parts, and place a copy of the return reference inside. Photograph the packed contents before sealing.",
     "A returned desk lamp arrives without its power adapter. The return team contacts the customer before deciding whether the assessment can proceed.", "What should I include when packing a return?"),
    ("Return shipping costs", "The shop pays return shipping for an incorrect or damaged item. Customers pay return shipping for a change of mind.",
     "Support verifies the return reason before issuing a prepaid label. If the customer chooses a different carrier, reimbursement requires written approval obtained before posting. Keep the carrier receipt until the case closes.",
     "A customer receives the wrong size because the warehouse picked incorrectly. Support issues a prepaid return label for that item.", "Who pays postage when the shop sends the wrong item?"),
    ("Refund processing", "Approved refunds are initiated within 5 business days after the returned item passes inspection. The payment provider may take another 3 to 7 business days to post the credit.",
     "The warehouse records inspection completion and sends confirmation. Refunds go to the original payment method. Support can provide a refund reference after initiation but cannot shorten the bank's posting period.",
     "An inspection is approved on Monday. The shop initiates the refund within five business days; the customer's bank then applies its own posting time.", "How long does a refund take after inspection?"),
    ("Exchanging an item", "Unused items may be exchanged within 30 calendar days, subject to replacement stock. Exchanges require a receipt and completed inspection.",
     "Choose the replacement size or color before returning the original item. A higher-priced replacement requires payment of the difference; a lower-priced replacement produces a refund of the difference to the original method.",
     "A customer exchanges an unused shirt for a larger size at the same price. If stock is available, no price adjustment is needed.", "Can I exchange an unused shirt for another size?"),
    ("Damaged on arrival", "Report visible transit damage within 48 hours of delivery. Keep the packaging and provide photographs of the item, shipping label, and damaged carton.",
     "Stop using the item if it appears unsafe. Support reviews the evidence and arranges collection or replacement. Do not discard damaged goods until support confirms whether collection is required.",
     "A ceramic bowl arrives cracked. The customer sends photographs the next day and keeps the box while support arranges a replacement.", "What should I do if an item arrives broken?"),
    ("Incorrect items", "Report an item that differs from the order confirmation within 7 calendar days of delivery. The shop covers approved correction shipping.",
     "Compare the product code on the item with the order confirmation. Send both codes to support with a photograph. Support verifies the picking error and supplies instructions for exchanging the incorrect item.",
     "The confirmation lists a blue notebook, but a red notebook arrives. The customer reports the mismatch and receives correction instructions.", "How do I report receiving the wrong product?"),
    ("Return exclusions", "Personalized goods, opened hygiene products, and redeemed digital codes are excluded from change-of-mind returns. Verified defects are reviewed separately.",
     "Check the product category and whether customization or redemption has occurred. Explain the applicable exclusion before offering a standard return. Escalate a claimed defect for review rather than rejecting it solely by category.",
     "An engraved bottle cannot be returned just because the customer dislikes the message. A cracked bottle is assessed as a possible defect.", "Can I return a personalized item because I changed my mind?"),
    ("Gift returns", "An unused gift with a gift receipt may be returned within 30 calendar days of the original purchase. Approved gift returns receive store credit.",
     "Use the gift receipt reference to identify the original purchase. Do not disclose the purchaser's payment details to the recipient. Explain that the return window starts at purchase, not the date the gift was opened.",
     "A recipient returns an unused gift with a valid gift receipt. Once approved, the recipient receives store credit instead of cash.", "What payment do I receive for an approved gift return?"),
    ("Standard delivery", "Standard domestic delivery takes 3 to 5 business days after dispatch. The estimate excludes weekends and listed shop holidays.",
     "Check the dispatch email for the start of the delivery estimate. Preparation time comes before transit time. Track the shipment using the carrier link and contact support if the last estimated delivery day has passed.",
     "An order dispatched on a business Monday normally arrives within the next three to five business days, subject to carrier exceptions.", "When should a standard domestic shipment arrive?"),
    ("Express delivery", "Express domestic delivery takes 1 to 2 business days after dispatch where the checkout offers this service. Express shipping carries an additional fee.",
     "Confirm express availability for the delivery postal code at checkout. Selecting express affects transit speed but does not bypass a product's stated preparation time. The order summary shows the extra shipping charge.",
     "An in-stock item is dispatched on Tuesday using express service. Its expected transit time is one to two business days.", "How quickly does express shipping arrive after dispatch?"),
    ("Free delivery threshold", "Standard domestic delivery is free when the merchandise subtotal after discounts is at least INR 999. Express and international delivery are excluded.",
     "Calculate the qualifying subtotal after applying discounts and before adding shipping. Gift-card payment does not reduce the merchandise subtotal. If the order falls below the threshold, checkout displays the applicable standard shipping fee.",
     "An order worth INR 1,100 receives a discount of INR 150. The resulting INR 950 subtotal does not qualify for free standard delivery.", "Does a discounted subtotal of INR 950 qualify for free shipping?"),
    ("Dispatch preparation", "In-stock orders normally require 1 to 2 business days for preparation before dispatch. Made-to-order products follow the preparation estimate on their product page.",
     "Distinguish order confirmation from dispatch confirmation. Payment checks and address clarification can pause preparation. The customer receives tracking information only after the parcel is assigned to a carrier.",
     "A customer places an in-stock order on Friday evening. Weekend days do not count toward the one-to-two-business-day preparation estimate.", "Is order preparation included in the delivery transit estimate?"),
    ("Changing an address", "A delivery address may be changed before the order reaches the dispatched status. Changes after dispatch depend on the carrier and are not guaranteed.",
     "Contact support immediately with the order number and corrected address. Support verifies the requester and checks shipment status before making changes. Do not assume that editing the account address changes an existing order.",
     "A customer updates the address book after placing an order. The current order still uses its original address until support confirms a change.", "Does editing my saved address update an order already placed?"),
    ("Tracking a shipment", "Tracking links are emailed after dispatch. Carrier tracking events may take up to 24 hours to appear after the first dispatch notification.",
     "Open the tracking link from the dispatch message and check its latest event time. A label-created status does not prove a parcel is moving. Contact support if no carrier event appears after 24 hours.",
     "A parcel receives a tracking number at 6 PM, but the carrier page remains empty that evening. The first scan may appear the following day.", "How long can it take for tracking events to appear?"),
    ("Missed delivery", "The carrier normally makes two delivery attempts. After unsuccessful attempts, the parcel may be held locally or returned to the shop.",
     "Read the carrier's missed-delivery message for rescheduling options. Provide a reachable phone number and accurate access instructions. Support can help interpret tracking but cannot guarantee a third delivery attempt.",
     "A customer misses the first attempt and schedules the second using the carrier link. The customer ensures someone is available to receive the parcel.", "How many delivery attempts are normally made?"),
    ("Lost shipments", "Ask support to investigate if tracking has not changed for 5 business days or shows delivered while the parcel cannot be located.",
     "Check with household members and the building reception before opening a case. Support requests delivery evidence from the carrier. A replacement or refund is decided after investigation, not automatically from a missing scan.",
     "Tracking says delivered, but neither the customer nor reception received the package. Support opens an investigation with the carrier.", "What should I do if tracking says delivered but I have no parcel?"),
    ("International delivery", "International delivery takes 7 to 14 business days after dispatch to supported countries. Customs review may add time beyond this estimate.",
     "Check country availability at checkout before placing the order. Supply a complete address and reachable telephone number. Customs-related requests come from the carrier or relevant authority and should be reviewed promptly.",
     "An overseas parcel reaches customs during the estimated transit window. Clearance delays can extend its arrival beyond fourteen business days.", "What is the estimated international delivery time?"),
    ("Customs charges", "For international orders, the recipient pays import duties and local clearance charges unless checkout explicitly states that these charges are included.",
     "Review the checkout breakdown and retain it with the order confirmation. The shop cannot provide a universal duty estimate for every destination. A refusal to pay customs charges can lead to a returned shipment.",
     "Checkout lists shipping but no prepaid import duties. The recipient may receive a separate duty request from the carrier.", "Who pays import duties if checkout does not include them?"),
    ("Supported payments", "The shop accepts credit cards, debit cards, and UPI for domestic orders. Cash on delivery is not available.",
     "Choose an available payment method at checkout and complete the provider's verification. Availability can vary by destination. Support never asks customers to reveal full card numbers, PINs, or one-time passwords.",
     "A domestic customer pays by UPI and receives confirmation after the payment provider reports success. Paying cash to the delivery driver is not an option.", "Can I pay cash when my parcel arrives?"),
    ("Failed payments", "A failed checkout does not create a confirmed paid order. Customers should check their order history before attempting payment again.",
     "If money was debited without confirmation, keep the payment reference and contact support. Avoid repeated attempts until status is checked. The payment provider determines when an unsuccessful debit is reversed.",
     "A UPI transaction shows a debit, but no order confirmation arrives. Support checks the reference before recommending another payment attempt.", "What if payment was debited but no order was confirmed?"),
    ("Duplicate charges", "Report a suspected duplicate payment with the order number and both transaction references. Support checks whether the entries are settled charges or temporary authorizations.",
     "Compare amounts, dates, and transaction identifiers rather than relying only on notifications. Do not send full payment credentials. Confirmed duplicate settlements are referred for correction through the original payment provider.",
     "Two notifications appear for one order, but one entry is pending. Support verifies settlement before labeling it a duplicate charge.", "What evidence helps investigate a duplicate payment?"),
    ("Invoices", "A digital invoice is available in the account after dispatch. Billing details should be entered correctly before the order is confirmed.",
     "Open order history and select the invoice download. Contact support if the invoice is missing or contains an error. A request to change legal billing details requires review and may not be possible after issuance.",
     "A customer needs an invoice for reimbursement. After dispatch, the customer downloads the digital invoice from the relevant order.", "When can I download an invoice?"),
    ("Discount codes", "Only one promotional discount code may be used per order. Codes must be entered before payment and cannot be added retroactively.",
     "Check the code's expiry, eligible products, and minimum subtotal. An automatic offer may make another code unavailable. Review the final total before paying because eligibility is assessed at checkout.",
     "A customer forgets to apply a promotion before payment. Support cannot attach the code to the completed order afterward.", "Can I add a discount code after paying?"),
    ("Gift cards", "Shop gift cards remain valid for 12 months from issue and can be used across multiple orders until their balance is exhausted.",
     "Enter the gift-card code at checkout and check the remaining balance in the account. If the balance is insufficient, use another supported method for the remainder. Treat the code like cash and keep it private.",
     "A gift card has INR 600 remaining and the order totals INR 800. The customer pays the additional INR 200 with a supported payment method.", "Can I split payment between a gift card and another method?"),
    ("Store credit", "Store credit is attached to the customer account and expires 6 months after issue. It cannot be transferred to another account or exchanged for cash.",
     "Sign in to the account that received the credit before checkout. Review the expiry date and available balance. For a return paid partly with credit, the credited portion returns to the account under the applicable expiry terms.",
     "A customer tries to use a friend's store credit. The balance is unavailable because it belongs to a different account.", "Can store credit be transferred to a friend?"),
    ("Order cancellation", "Orders may be canceled before dispatch. Once dispatched, the customer must follow the return process instead of requesting cancellation.",
     "Use the cancel option in order history if it is available. Otherwise contact support with the order number. A cancellation request is not complete until confirmation is received; processing can continue while a request is being checked.",
     "A customer requests cancellation after the carrier collects the parcel. Support explains that the shipment now follows the return process.", "Can an order be canceled after dispatch?"),
    ("Editing an order", "Products, sizes, and quantities cannot be edited after order confirmation. A customer may cancel an eligible undispatched order and place a new one.",
     "Check dispatch status before requesting cancellation. Prices and stock are checked again for the replacement order. Support cannot reserve the original promotional price while a different order is being placed.",
     "A customer orders two notebooks but wants three. If cancellation is still available, the customer can cancel and submit a new order for three.", "Can I change the quantity after confirming an order?"),
    ("Preorders", "Preorder product pages state an estimated dispatch date. An order containing a preorder may be held until that product is ready unless split shipment is offered.",
     "Read the preorder notice before checkout and retain the confirmation. Estimated dates can change; support sends an update when a material delay is known. Cancellation eligibility is checked against dispatch status.",
     "A basket contains an available pen and a preorder planner. If split shipment is not offered, both may ship when the planner is ready.", "Can an in-stock item wait for a preorder in the same order?"),
    ("Out-of-stock alerts", "Customers may request an email alert for an unavailable product. An alert is a notification, not a reservation or guarantee of stock.",
     "Choose the correct size and color before subscribing to an alert. When the notification arrives, visit the product page to check availability. Popular stock may sell out before the customer completes checkout.",
     "A customer receives a restock email but returns two days later. The alert does not guarantee that the item is still available.", "Does a restock alert reserve a product for me?"),
    ("Split shipments", "An order may arrive in multiple parcels when items are dispatched from different locations. Each parcel receives its own tracking reference.",
     "Compare the items listed in each dispatch email with the full order. Receiving one parcel does not mean the entire order was delivered. Contact support only after checking whether another shipment remains in transit.",
     "A customer orders a lamp and a notebook. The notebook arrives first, and a second tracking reference shows the lamp is still on its way.", "Why did only part of my order arrive?"),
    ("Quantity limits", "Some promotional products have a limit of two units per customer. The product page states any applicable limit before checkout.",
     "Review product restrictions when ordering multiple units. Orders designed to bypass a published quantity limit may be reviewed or canceled. Contact the bulk-order team for legitimate larger requirements.",
     "A limited promotion allows two bottles per customer. Placing separate orders does not create an entitlement to exceed the published limit.", "Can I bypass a purchase limit with separate orders?"),
    ("Bulk orders", "Requests for 20 or more units should be sent to the bulk-order team for a written quote. Standard promotional codes do not automatically apply to quoted orders.",
     "Provide product codes, quantities, destination, and required date. The quote identifies pricing, payment terms, and delivery assumptions. A quote does not reserve inventory until the stated acceptance conditions are met.",
     "A business needs 35 notebooks for an event. It requests a bulk quote instead of assuming the retail checkout can guarantee that quantity.", "How should I order 35 units for an event?"),
    ("Price adjustments", "The shop does not provide retrospective price adjustments when a product price drops after a completed purchase.",
     "Review the price shown at checkout before payment. A later promotion applies to new qualifying purchases under its own terms. Support can explain the original invoice but cannot automatically refund a later price difference.",
     "A bag goes on sale a week after purchase. The earlier order retains the price accepted when it was placed.", "Can I claim a price difference after a later sale?"),
    ("Product sizing", "Product-specific size charts should be used before ordering. Measurements may differ across brands and product ranges.",
     "Measure according to the guide on the relevant product page. Compare the measurement with that product's chart rather than a previous purchase from another brand. Contact support if a listed dimension is unclear.",
     "A customer usually buys a medium in another brand. The new product's measurements indicate a large, so the customer follows its own size chart.", "Should I use one brand's size chart for another brand?"),
    ("Care instructions", "Follow the care label supplied with each product. Damage caused by cleaning methods that contradict the label is not treated as a manufacturing defect.",
     "Check washing temperature, drying method, and restrictions before first use. Test a permitted cleaning method on a small area when appropriate. Support can help locate missing instructions but does not replace product-specific guidance.",
     "A fabric pouch marked hand-wash-only is damaged in a hot machine cycle. This is assessed as care-related damage rather than a manufacturing fault.", "Does damage from ignoring the care label count as a defect?"),
    ("Warranty coverage", "Products marked with the shop warranty have 12 months of coverage for manufacturing defects from the purchase date. Accidental damage and normal wear are excluded.",
     "Keep the invoice and product serial number where applicable. Describe when the fault first appeared and whether the item has been modified. Warranty assessment may require inspection before repair or replacement is approved.",
     "A covered lamp stops working after normal use at six months. Support opens a manufacturing-defect assessment using the invoice and serial number.", "How long is the shop's manufacturing-defect warranty?"),
    ("Making a warranty claim", "Warranty claims require proof of purchase, a description of the fault, and clear photographs or video where the issue can be shown safely.",
     "Stop using an electrical product if it appears unsafe. Submit evidence through support and wait for inspection instructions. Do not arrange an unauthorized repair before the claim is assessed, as this may affect coverage.",
     "A customer notices an intermittent fault and records a safe demonstration. Support reviews the evidence before deciding whether to collect the product.", "What evidence is needed for a warranty claim?"),
    ("Repair turnaround", "An approved workshop repair is normally completed within 10 business days after the workshop receives the item. Parts shortages may extend the estimate.",
     "The workshop confirms receipt separately from the customer's shipping receipt. Support provides an update if parts are unavailable. Return transit time is additional to the workshop's repair estimate.",
     "An item reaches the workshop on Wednesday. The repair estimate begins at workshop receipt rather than when the customer posted the parcel.", "When does the repair turnaround estimate begin?"),
    ("Spare parts", "Spare-part availability depends on the product model. Support checks compatibility before quoting a replacement accessory or component.",
     "Provide the product code, serial number if present, and the part required. Do not order a visually similar part without confirmation. Installation guidance is limited to the approved instructions for that product.",
     "Two lamps use similar-looking adapters with different specifications. Support checks the exact lamp model before recommending an adapter.", "How can I find a compatible spare part?"),
    ("Support opening hours", "Customer support is available Monday to Friday from 9 AM to 6 PM India Standard Time, excluding listed shop holidays.",
     "Messages received outside opening hours enter the next available business-day queue. Include the order number and a concise description in the first message. A delivery carrier may operate on a different schedule.",
     "A customer emails on Saturday evening. The message is queued for the next support business day rather than receiving a guaranteed weekend response.", "When is customer support available?"),
    ("Support response target", "The support team aims to send an initial response within 1 business day. This is an acknowledgment target, not a guarantee that the issue is resolved in that time.",
     "Use one support ticket per issue and reply in the same thread. Include requested evidence to reduce follow-up messages. Complex carrier or payment investigations can require additional time after the first response.",
     "Support acknowledges a missing parcel within one business day, then waits for the carrier's investigation before giving a final decision.", "Does the first-response target mean my problem will be resolved that day?"),
    ("Escalating a complaint", "A customer may request a supervisor review after receiving the first support decision. The review target is 3 business days after all requested evidence is available.",
     "Quote the existing ticket number and explain the disputed point. Include new evidence rather than opening duplicate tickets. The supervisor reviews the recorded facts and communicates the outcome in the same case thread.",
     "A customer disagrees with a return assessment and supplies a missing photograph. The supervisor's review clock starts after the evidence is complete.", "How long does a supervisor review normally take?"),
    ("Accessible shopping support", "Customers who need help using the website may request an assisted shopping conversation through support. Staff can explain navigation and product information.",
     "Describe the access difficulty and the preferred contact method. Staff do not collect payment passwords or one-time codes. The customer completes payment through the approved checkout or payment provider.",
     "A customer has difficulty locating a size guide with a screen reader. Support explains where the information is available and reads the relevant measurements.", "Can support help me navigate product information?"),
    ("Account recovery", "Account recovery uses the registered email address and the password-reset process. Support cannot disclose an existing password.",
     "Request a reset link and check the spam folder if it does not arrive. If access to the registered email is lost, contact support for account-ownership review. Do not share a reset link with another person.",
     "A customer forgets a password. Support directs the customer to reset it instead of asking for or revealing the old password.", "Will support tell me my existing password?"),
    ("Changing contact details", "Changes to the registered email or mobile number may require verification. Account changes do not automatically alter contact details on dispatched orders.",
     "Update the account profile and complete the requested verification. For an active shipment, contact support separately and supply its order number. Carrier contact details can remain unchanged if the shipment cannot be edited.",
     "A customer changes their mobile number in the account while a parcel is in transit. Support checks whether the carrier can also update its contact record.", "Does changing my account phone number update an active shipment?"),
    ("Privacy requests", "Customers may submit a request to access or delete their account information through support. Identity verification is required before a request is processed.",
     "State the request clearly and use the registered contact address where possible. Support explains which records can be removed and which transaction records may need to be retained under applicable obligations. Do not send unnecessary identity documents.",
     "A customer asks to delete an account. Support verifies the requester and explains the handling of account data separately from retained transaction records.", "What happens before an account-data deletion request is processed?"),
    ("Product reviews", "Reviews should describe a genuine product experience and must not include private contact details, payment information, or abusive content.",
     "Use the review form associated with the product and keep order disputes in the support channel. Reviews may be moderated for content-rule violations. A critical opinion is not removed solely because it is negative.",
     "A customer posts useful criticism but includes a phone number. Moderation removes the private detail or requests a revised submission.", "Can a review include my phone number?"),
    ("Holiday service calendar", "The fictional shop closes support and dispatch operations on 1 January, 15 August, and 2 October. These dates do not count as shop business days.",
     "When estimating a dispatch or support target, exclude weekends and these listed closures. Carrier and bank holidays can differ. Recalculate the estimate if the relevant external provider has an additional non-working day.",
     "A two-business-day support process crosses a listed shop holiday. The closed date is skipped when calculating the target completion day.", "Which dates are excluded from the shop's business-day calendar?"),
]


def main():
    if len(TOPICS) != 50:
        raise ValueError(f"Expected 50 distinct topics, found {len(TOPICS)}")
    output = Path(__file__).with_name("sample_document.pdf")
    navy, teal = colors.HexColor("#152B3C"), colors.HexColor("#167D8D")
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=navy, spaceAfter=16),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=teal, spaceBefore=17, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=11.5, leading=17, textColor=navy, spaceAfter=8),
        "rule": ParagraphStyle("rule", fontName="Helvetica-Bold", fontSize=12, leading=18, textColor=navy),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#526779")),
    }
    def para(text, kind="body"):
        return Paragraph(escape(text), styles[kind])

    def page_frame(canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(navy)
        canvas.rect(0, height - 13, width, 13, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(48, height - 43, "NORTHSTAR SHOP | PRACTICE HANDBOOK")
        canvas.setStrokeColor(colors.HexColor("#D9E3E9"))
        canvas.line(48, 51, width - 48, 51)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526779"))
        canvas.drawString(48, 36, "FICTIONAL TRAINING DATA - NOT A REAL SHOP POLICY")
        canvas.drawRightString(width - 48, 36, f"{doc.page} / 50")
        canvas.restoreState()

    story = []
    for number, (title, rule, procedure, example, question) in enumerate(TOPICS, start=1):
        story.append(para(f"POLICY {number:02d} / 50", "section"))
        story.append(para(title, "title"))
        box = Table([[para(rule, "rule")]], colWidths=[A4[0] - 96])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF5F6")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C6E0E3")),
            ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ]))
        story.append(box)
        for heading, body in (("HOW TO HANDLE THE REQUEST", procedure), ("WORKED EXAMPLE", example),
                              ("PRACTICE QUESTION", question)):
            story.extend([para(heading, "section"), para(body)])
        story.extend([
            para("CHECK YOUR ANSWER", "section"),
            para(f"Use the rule and worked example on this page to answer the practice question. "
                 f"Retain any stated deadline, eligibility condition, or exception. Cite PDF page {number} "
                 "so another reader can verify the source. If a requested detail is absent, say that the document does not specify it."),
            Spacer(1, 14),
            para("This handbook is synthetic learning material for PDF retrieval experiments. "
                 "Its invented procedures are not legal advice or the terms of an actual business.", "small"),
        ])
        if number < len(TOPICS):
            story.append(PageBreak())
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48,
                            topMargin=65, bottomMargin=65, title="Northstar Shop - 50 Policy Lessons",
                            author="Semantic RAG Practice Project")
    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    reader = PdfReader(output)
    if len(reader.pages) != 50:
        raise ValueError(f"Layout produced {len(reader.pages)} pages; expected exactly 50.")
    for number, page in enumerate(reader.pages, start=1):
        if len((page.extract_text() or "").split()) < 100:
            raise ValueError(f"Page {number} has unexpectedly little extractable content.")
    print(f"Created {output}: 50 pages with selectable text.")


if __name__ == "__main__":
    main()
