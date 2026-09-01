"""Import every ORM model here so `Base.metadata` is complete for Alembic
autogenerate and for `Base.metadata.create_all()` in tests. Add new model
modules to this list as phases land — nothing else needs to change."""
from app.db.base import Base  # noqa: F401
from app.models.annotation import (  # noqa: F401
    Annotation,
    AnnotationEvent,
    AnnotationEventAction,
    AnnotationReviewStatus,
    AnnotationSource,
    ErrorCategory,
    ErrorReason,
)
from app.models.blob_import_job import BlobImportJob, BlobImportJobStatus  # noqa: F401
from app.models.dataset import Dataset, DatasetStatus  # noqa: F401
from app.models.dataset_version import (  # noqa: F401
    DatasetVersion,
    DatasetVersionAnnotationPin,
    DatasetVersionImage,
    DatasetVersionStatus,
    SplitName,
)
from app.models.image import Image, ImageReviewStatus, ImageSourceType  # noqa: F401
from app.models.inference_job import InferenceJob, JobStatus  # noqa: F401
from app.models.integration import Integration, IntegrationProvider  # noqa: F401
from app.models.ml_model import MLModel, ModelKind  # noqa: F401
from app.models.project import Project  # noqa: F401
from app.models.quality import AnnotationFlag, FlagResolution, FlagType, ImagePoseContext  # noqa: F401
from app.models.roboflow_job import RoboflowJob, RoboflowJobKind, RoboflowJobStatus  # noqa: F401
from app.models.training_job import (  # noqa: F401
    TrainingJob,
    TrainingJobEpoch,
    TrainingJobStatus,
    TrainingProviderType,
)
from app.models.video import Video, VideoStatus  # noqa: F401

__all__ = [
    "Base",
    "Project",
    "Dataset",
    "DatasetStatus",
    "Video",
    "VideoStatus",
    "Image",
    "ImageSourceType",
    "ImageReviewStatus",
    "MLModel",
    "ModelKind",
    "Annotation",
    "AnnotationEvent",
    "AnnotationEventAction",
    "AnnotationSource",
    "AnnotationReviewStatus",
    "ErrorCategory",
    "ErrorReason",
    "InferenceJob",
    "JobStatus",
    "Integration",
    "IntegrationProvider",
    "DatasetVersion",
    "DatasetVersionStatus",
    "DatasetVersionImage",
    "DatasetVersionAnnotationPin",
    "SplitName",
    "TrainingJob",
    "TrainingJobEpoch",
    "TrainingJobStatus",
    "TrainingProviderType",
    "AnnotationFlag",
    "FlagType",
    "FlagResolution",
    "ImagePoseContext",
    "RoboflowJob",
    "RoboflowJobKind",
    "RoboflowJobStatus",
    "BlobImportJob",
    "BlobImportJobStatus",
]
