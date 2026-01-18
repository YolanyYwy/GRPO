"""Data models for the delivery domain."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from AGentCL.environment.db import DB


class Location(BaseModel):
    """Location information."""

    address: str = Field(description="Address")
    longitude: float = Field(description="Longitude")
    latitude: float = Field(description="Latitude")

    def __repr__(self):
        return f"Location(address={self.address}, longitude={self.longitude}, latitude={self.latitude})"


class StoreProduct(BaseModel):
    """Product in a store."""

    product_id: str = Field(description="Product ID")
    name: str = Field(description="Product name")
    store_id: str = Field(description="Store ID")
    store_name: str = Field(description="Store name")
    price: float = Field(description="Product price")
    quantity: int = Field(default=1, description="Product quantity")
    attributes: str = Field(default="", description="Product attributes")
    tags: List[str] = Field(default_factory=list, description="Product tags")

    def __repr__(self):
        return (f"StoreProduct(store_name={self.store_name}, "
                f"store_id={self.store_id}, "
                f"product_name={self.name}, "
                f"product_id={self.product_id}, "
                f"attributes={self.attributes}, "
                f"quantity={self.quantity}, "
                f"price={self.price}, "
                f"tags={self.tags})")


class Store(BaseModel):
    """Store information."""

    store_id: str = Field(description="Store ID")
    name: str = Field(description="Store name")
    score: float = Field(description="Store rating")
    location: Location = Field(description="Store location")
    tags: List[str] = Field(description="Store tags")
    products: List[StoreProduct] = Field(description="List of products")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Store(name={self.name}, "
                f"store_id={self.store_id}, "
                f"score={self.score}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Store(name={self.name}, "
                f"store_id={self.store_id}, "
                f"score={self.score}, "
                f"location={repr(self.location)}, "
                f"tags={self.tags})")


class Order(BaseModel):
    """Order information."""

    order_id: str = Field(description="Order ID")
    order_type: str = Field(description="Order type")
    user_id: str = Field(description="User ID")
    store_id: str = Field(description="Store ID")
    location: Location = Field(description="Delivery location")
    dispatch_time: str = Field(description="Dispatch time")
    shipping_time: float = Field(description="Shipping time in minutes")
    delivery_time: str = Field(description="Delivery time")
    total_price: float = Field(description="Total price")
    create_time: str = Field(description="Order creation time")
    update_time: str = Field(description="Order update time")
    note: str = Field(default="", description="Order note")
    products: List[StoreProduct] = Field(description="Ordered products")
    status: str = Field(description="Order status")

    def __repr__(self):
        products_repr = "\n".join(repr(p) for p in self.products)
        return (f"Order(order_id={self.order_id}, "
                f"order_type={self.order_type}, "
                f"user_id={self.user_id}, "
                f"store_id={self.store_id}, "
                f"location={repr(self.location)}, "
                f"dispatch_time={self.dispatch_time}, "
                f"shipping_time={self.shipping_time}, "
                f"delivery_time={self.delivery_time}, "
                f"total_price={self.total_price}, "
                f"create_time={self.create_time}, "
                f"update_time={self.update_time}, "
                f"note={self.note}, "
                f"status={self.status}, "
                f"products=[\n{products_repr}\n])")

    def __str__(self):
        return (f"Order(order_id={self.order_id}, "
                f"order_type={self.order_type}, "
                f"user_id={self.user_id}, "
                f"store_id={self.store_id}, "
                f"total_price={self.total_price}, "
                f"create_time={self.create_time}, "
                f"status={self.status})")


class DeliveryDB(DB):
    """Database for the delivery domain."""

    stores: Dict[str, Store] = Field(
        default_factory=dict,
        description="Dictionary of stores indexed by store ID"
    )
    orders: Dict[str, Order] = Field(
        default_factory=dict,
        description="Dictionary of orders indexed by order ID"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics of the database."""
        return {
            "num_stores": len(self.stores),
            "num_orders": len(self.orders),
        }
