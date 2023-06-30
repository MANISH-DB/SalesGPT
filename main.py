from typing import List, Any

from fastapi import FastAPI
from langchain.schema import Document
from pydantic import BaseModel, AnyHttpUrl

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class YoovChatbotProduct(BaseModel):
    id: str
    name: str
    type: str
    original_price: float | None = None
    sale_price: float | None = None
    short_description: str | None = None
    description: str | None = None
    cover_image: AnyHttpUrl | None = None
    images: List[AnyHttpUrl]
    categories: List[str]
    attributes: dict[str, Any]
    variations: List[Any]


def create_content(product: YoovChatbotProduct):
    print(product)
    return f"""product name: {product.name}
    product type: {product.type}
    product original price: {product.original_price}
    product sale price: {product.sale_price}
    product description: {product.description}
    product attributes: {product.attributes}
"""


def create_documents_from_yoov_products(products: List[YoovChatbotProduct]):
    documents = []
    for item in products:
        document = Document(
            page_content=create_content(item),
            metadata={
                "product_id": item.id,
                'cover_image': item.cover_image,
                'images': item.images
            },
        )

        if len(item.variations) > 0:
            vs = []
            for v in item.variations:
                product = YoovChatbotProduct(**v)
                vs.append(f"{{{create_content(product)}}}")

            document.page_content += f"product variations: {vs}"

        documents.append(document)

    return documents


@app.post("/woocommerce/products")
async def create_item(products: List[YoovChatbotProduct]):
    documents = create_documents_from_yoov_products(products)
    return documents
