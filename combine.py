import qrcode
from PIL import Image
data = "https://www.google.com/search?q=sc&rlz=1C5CHFA_enAU848AU848&oq=sc&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARBFGDsyBggCEEUYOzIGCAMQRRg8MgYIBBBFGDwyBggFEEUYPDIGCAYQRRg9MgYIBxBFGDzSAQcxOThqMGo5qAIAsAIA&sourceid=chrome&ie=UTF-8"  # Replace with your data
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(data)
qr.make(fit=True)

# Create a QR code image
qr_code = qr.make_image(fill_color="black", back_color="white")
qr_code.save('qrcode.png')