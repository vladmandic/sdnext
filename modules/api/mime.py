import mimetypes


def register():
    mimetypes.init()
    mimetypes.add_type('application/javascript', '.js')
    mimetypes.add_type('application/javascript', '.mjs')
    mimetypes.add_type('application/json', '.map')
    mimetypes.add_type('text/html', '.html')
    mimetypes.add_type('image/webp', '.webp')
    mimetypes.add_type('image/jxl', '.jxl')
    mimetypes.add_type('font/ttf', '.ttf')
