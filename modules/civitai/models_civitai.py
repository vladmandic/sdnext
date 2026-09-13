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
    hash: str | None = None
    meta: dict | None = None


class CivitFileHashes(BaseModel):
    class Config:
        allow_population_by_field_name = True
    sha256: str | None = Field(None, alias="SHA256")
    autov1: str | None = Field(None, alias="AutoV1")
    autov2: str | None = Field(None, alias="AutoV2")
    autov3: str | None = Field(None, alias="AutoV3")
    crc32: str | None = Field(None, alias="CRC32")
    blake3: str | None = Field(None, alias="BLAKE3")
    sha256_12: str | None = Field(None, alias="SHA256_12")


class CivitFileMetadata(BaseModel):
    class Config:
        allow_population_by_field_name = True
    format: str | None = None
    size: str | None = None
    fp: str | None = None
    quant_type: str | None = Field(None, alias="quantType")


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
    metadata: CivitFileMetadata = Field(default_factory=CivitFileMetadata)
    pickle_scan_result: str | None = Field(None, alias="pickleScanResult")
    virus_scan_result: str | None = Field(None, alias="virusScanResult")
    scanned_at: str | None = Field(None, alias="scannedAt")


class CivitStats(BaseModel):
    # counts CivitAI does not report arrive as null, not zero
    class Config:
        allow_population_by_field_name = True
    download_count: int | None = Field(0, alias="downloadCount")
    thumb_up_count: int | None = Field(0, alias="thumbsUpCount")
    thumb_down_count: int | None = Field(0, alias="thumbsDownCount")
    comment_count: int | None = Field(0, alias="commentCount")
    tipped_amount_count: int | None = Field(0, alias="tippedAmountCount")


class CivitVersion(BaseModel):
    class Config:
        allow_population_by_field_name = True
    id: int = 0
    model_id: int = Field(0, alias="modelId")
    name: str = "Unknown"
    base_model: str = Field("Unknown", alias="baseModel")
    published_at: str | None = Field(None, alias="publishedAt")
    availability: str = "Unknown"
    early_access_ends_at: str | None = Field(None, alias="earlyAccessEndsAt")
    early_access_config: dict | None = Field(None, alias="earlyAccessConfig")
    description: str | None = None
    trained_words: list[str] = Field(default_factory=list, alias="trainedWords")
    stats: CivitStats = Field(default_factory=CivitStats)
    files: list[CivitFile] = Field(default_factory=list)
    images: list[CivitImage] = Field(default_factory=list)
    nsfw_level: int = Field(0, alias="nsfwLevel")
    download_url: str = Field("", alias="downloadUrl")

    @validator('availability', pre=True)
    def coerce_null_availability(cls, v): # pylint: disable=no-self-argument
        # /model-versions/{id} serializes availability as null, which failed
        # str validation and turned every version lookup into a 404. Fall back
        # to the default instead of rejecting the whole version.
        return "Unknown" if v in (None, "") else v


class CivitVersionMini(BaseModel):
    # primary file flattened onto the version plus permission flags; earlyAccessEndsAt and freeTrialLimit exist only during early access
    class Config:
        allow_population_by_field_name = True
    air: str = ""
    version_name: str = Field("", alias="versionName")
    model_name: str = Field("", alias="modelName")
    user_id: int = Field(0, alias="userId")
    base_model: str = Field("Unknown", alias="baseModel")
    availability: str = "Unknown"
    published_at: str | None = Field(None, alias="publishedAt")
    size: float = 0
    file_type: str = Field("", alias="fileType")
    file_name: str = Field("", alias="fileName")
    format: str = ""
    hashes: CivitFileHashes = Field(default_factory=CivitFileHashes)
    download_urls: list[str] = Field(default_factory=list, alias="downloadUrls")
    can_generate: bool = Field(False, alias="canGenerate")
    is_featured: bool = Field(False, alias="isFeatured")
    require_auth: bool = Field(False, alias="requireAuth")
    check_permission: bool = Field(False, alias="checkPermission")
    additional_resource_charge: bool = Field(False, alias="additionalResourceCharge")
    payout_enabled: bool = Field(False, alias="payoutEnabled")
    minor: bool = False
    sfw_only: bool = Field(False, alias="sfwOnly")
    fees: list = Field(default_factory=list)
    early_access_ends_at: str | None = Field(None, alias="earlyAccessEndsAt")
    free_trial_limit: int | None = Field(None, alias="freeTrialLimit")


class CivitCreator(BaseModel):
    class Config:
        allow_population_by_field_name = True
    username: str = "Unknown"
    image: str | None = None

    @validator('username', pre=True)
    def coerce_null_username(cls, v): # pylint: disable=no-self-argument
        # Deleted/anonymous creators serialize username as null, which failed
        # str validation and rejected the entire search response. Fall back to
        # the default name instead of dropping the whole page.
        return "Unknown" if v in (None, "") else v


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
    supports_generation: bool = Field(False, alias="supportsGeneration")
    mode: str | None = None
    poi: bool = False
    minor: bool = False
    sfw_only: bool = Field(False, alias="sfwOnly")
    user_id: int = Field(0, alias="userId")
    cosmetic: dict | None = None

    @validator('allow_commercial_use', pre=True)
    def coerce_commercial_use(cls, v): # pylint: disable=no-self-argument
        if isinstance(v, str):
            return [v] if v else []
        return v

    @validator('poi', 'minor', 'sfw_only', pre=True)
    def coerce_null_flags(cls, v): # pylint: disable=no-self-argument
        # These flags serialize as null on some endpoints; treat null as not-set.
        return False if v is None else v


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
    error: str | None = None  # server or parse failure text; items is empty when set


class CivitTag(BaseModel):
    class Config:
        allow_population_by_field_name = True
    name: str = ""
    model_count: int = Field(0, alias="modelCount")
    link: str = ""


class CivitTagResponse(BaseModel):
    class Config:
        allow_population_by_field_name = True
    items: list[CivitTag] = Field(default_factory=list)
    metadata: CivitSearchMetadata = Field(default_factory=CivitSearchMetadata)


class CivitCreatorItem(BaseModel):
    class Config:
        allow_population_by_field_name = True
    username: str = ""
    model_count: int = Field(0, alias="modelCount")
    link: str = ""
    image: str | None = None


class CivitCreatorResponse(BaseModel):
    class Config:
        allow_population_by_field_name = True
    items: list[CivitCreatorItem] = Field(default_factory=list)
    metadata: CivitSearchMetadata = Field(default_factory=CivitSearchMetadata)


class CivitUserProfile(BaseModel):
    # tier is omitted for non-members; email, emailVerified and tokenScope are left unmodelled to keep the address out of the API response
    class Config:
        allow_population_by_field_name = True
    id: int = 0
    username: str = ""
    tier: str | None = None
    status: str | None = None
    is_member: bool = Field(False, alias="isMember")
    subscriptions: list = Field(default_factory=list)
