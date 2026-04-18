from pinecone import Pinecone, ServerlessSpec
from askhr.config import get_settings



def get_index():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

    # exists = [idx.name for idx in pc.list_indexes()]
    # if settings.pinecone_index_name not in exists:
    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region
            )
        )
    return pc.Index(settings.pinecone_index_name)