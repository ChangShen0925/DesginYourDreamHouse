from PIL import Image
def compress(path):
    original_image = Image.open(f'{path}')
    width, height = 512, 512
    compressed_image = original_image.resize((width, height), Image.ANTIALIAS)
    compressed_image.save(f'{path}')

compress('/Users/changshen/Desktop/DesignYourDreamHouse/gothic .png')




