import os
import tempfile
from pathlib import Path
from pydantic import BaseModel
from fastapi import Request, Header, UploadFile, Form
from fastapi.exceptions import HTTPException
from modules import paths
from modules.logger import log
from modules.images import FilenameGenerator


"""
new endpoint: /sdapi/v1/upload
- if path is not given, file fill be uploaded to system temp folder
- if path is given, its considered as relative to sdnext root (datadir) and must exist
- absolute paths or paths outside of sdnext root are not allowed

example using post with formdata:
> curl -X POST "http://localhost:7860/sdapi/v1/upload" -F "file=@/home/vlado/dev/sdnext/config.json" -F overwrite=true -F path=data

example using put with raw bytes:
> curl -X PUT "http://localhost:7860/sdapi/v1/upload" -T config.json -H "filename:config.json" -H "path:data/" -H "overwrite:true"
"""

class ResUpload(BaseModel):
    input: str
    output: str
    mime: str
    size: int
    overwrite: bool


def check_file(filename, path, overwrite):
    namegen = FilenameGenerator()
    if len(path) > 0 and (os.path.isabs(path) or not os.path.isdir(path)):
        raise HTTPException(status_code=400, detail="Invalid path")
    fn = os.path.join(path, filename)
    fn = namegen.sanitize(fn)
    if Path(fn).parent == Path('.'): # just filename, no path
        fn = os.path.join(tempfile.gettempdir(), fn)
    else:
        fn = os.path.join(paths.data_path, fn)
    if os.path.exists(fn) and len(overwrite) == 0:
        raise HTTPException(status_code=400, detail="File exists")
    return fn

def put_upload(request: Request,
               filename: str = Header(''),
               filetype: str = Header('application/octet-stream'),
               overwrite: str = Header(''),
               path: str = Header('')
              ) -> ResUpload:
    fn = check_file(filename, path, overwrite)
    try:
        from asyncio import run
        content = run(request.body())
        with open(fn, 'wb') as f:
            f.write(content)
        res = ResUpload(input=filename, output=fn, mime=filetype, size=len(content), overwrite=len(overwrite) > 0)
        log.trace(f'API upload: {res.dict()}')
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail="Upload failed") from e

def post_upload(file: UploadFile, overwrite: str = Form(''), path: str = Form('')) -> ResUpload:
    fn = check_file(file.filename, path, overwrite)
    try:
        content = file.file.read()
        with open(fn, 'wb') as f:
            f.write(content)
        res = ResUpload(input=file.filename, output=fn, mime=file.content_type, size=len(content), overwrite=len(overwrite) > 0)
        log.trace(f'API upload: {res.dict()}')
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail="Upload failed") from e

def register_api():
    from modules.shared import api
    api.add_api_route("/sdapi/v1/upload", post_upload, methods=["POST"], response_model=ResUpload, tags=["Upload"])
    api.add_api_route("/sdapi/v1/upload", put_upload, methods=["PUT"], response_model=ResUpload, tags=["Upload"])
