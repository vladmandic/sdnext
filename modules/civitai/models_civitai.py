from pydantic import BaseModel, Field, validator # pylint: disable=no-name-in-module


class CivitImage(BaseModel):
    class Config:
        allow_population_by_field_name = True
    id: int = 0
    url: str = ""
    width: int = 0
    height: int = 0
    type: str = "Unknown"
    nsfw_level: int = Field(0, alias="nsfwLevel")


class CivitFileHashes(BaseModel):
    class Config:
        allow_population_by_field_name = True
    sha256: str | None = Field(None, alias="SHA256")
    autov1: str | None = Field(None, alias="AutoV1")
    autov2: str | None = Field(None, alias="AutoV2")
    autov3: str | None = Field(None, alias="AutoV3")
    crc32: str | None = Field(None, alias="CRC32")
    blake3: str | None = Field(None, alias="BLAKE3")


class CivitFile(BaseModel):
    class Config:
        allow_population_by_field_name = True
    id: int = 0
    name: str = "Unknown"
    type: str = "Unknown"
    size_kb: float = Field(0, alias="sizeKB")
    hashes: CivitFileHashes = Field(default_factory=CivitFileHashes)
    download_url: str = Field("", alias="downloadUrl")
    primary: bool | None = None


class CivitStats(BaseModel):
    class Config:
        allow_population_by_field_name = True
    download_count: int = Field(0, alias="downloadCount")
    favorite_count: int = Field(0, alias="favoriteCount")
    thumb_up_count: int = Field(0, alias="thumbsUpCount")
    thumb_down_count: int = Field(0, alias="thumbsDownCount")
    comment_count: int = Field(0, alias="commentCount")
    rating_count: int = Field(0, alias="ratingCount")
    rating: float = 0


class CivitVersion(BaseModel):
    class Config:
        allow_population_by_field_name = True
    id: int = 0
    model_id: int = Field(0, alias="modelId")
    name: str = "Unknown"
    base_model: str = Field("Unknown", alias="baseModel")
    published_at: str | None = Field(None, alias="publishedAt")
    availability: str = "Unknown"
    description: str | None = None
    trained_words: list[str] = Field(default_factory=list, alias="trainedWords")
    stats: CivitStats = Field(default_factory=CivitStats)
    files: list[CivitFile] = Field(default_factory=list)
    images: list[CivitImage] = Field(default_factory=list)
    nsfw_level: int = Field(0, alias="nsfwLevel")
    download_url: str = Field("", alias="downloadUrl")


class CivitCreator(BaseModel):
    class Config:
        allow_population_by_field_name = True
    username: str = "Unknown"
    image: str | None = None


class CivitModel(BaseModel):
    class Config:
        allow_population_by_field_name = True
    id: int = 0
    type: str = "Unknown"
    name: str = "Unknown"
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    nsfw: bool = False
    nsfw_level: int = Field(0, alias="nsfwLevel")
    availability: str = "Unknown"
    stats: CivitStats = Field(default_factory=CivitStats)
    creator: CivitCreator = Field(default_factory=CivitCreator)
    versions: list[CivitVersion] = Field(default_factory=list, alias="modelVersions")
    allow_no_credit: bool = Field(True, alias="allowNoCredit")
    allow_commercial_use: list[str] = Field(default_factory=list, alias="allowCommercialUse")
    allow_derivatives: bool = Field(True, alias="allowDerivatives")
    allow_different_license: bool = Field(True, alias="allowDifferentLicense")

    @validator('allow_commercial_use', pre=True)
    def _coerce_commercial_use(cls, v): # pylint: disable=no-self-argument
        if isinstance(v, str):
            return [v] if v else []
        return v


class CivitSearchMetadata(BaseModel):
    class Config:
        allow_population_by_field_name = True
    next_page: str | None = Field(None, alias="nextPage")
    current_page: int | None = Field(None, alias="currentPage")
    page_size: int | None = Field(None, alias="pageSize")
    total_pages: int | None = Field(None, alias="totalPages")
    total_items: int | None = Field(None, alias="totalItems")
    next_cursor: str | None = Field(None, alias="nextCursor")


class CivitSearchResponse(BaseModel):
    class Config:
        allow_population_by_field_name = True
    items: list[CivitModel] = Field(default_factory=list)
    metadata: CivitSearchMetadata = Field(default_factory=CivitSearchMetadata)
    request_url: str | None = Field(None, alias="requestUrl")
