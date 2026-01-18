# Delivery Domain Policy

## Overview
This policy governs the delivery domain operations including store search, product search, order creation, payment, and order management.

## Store Operations

1. **Store Search**
   - Use `delivery_store_search_recommand` to search for stores based on keywords
   - Consider store ratings, location, and tags when recommending stores
   - Guide users to select a specific store before proceeding with orders

2. **Store Information**
   - Use `get_delivery_store_info` to retrieve detailed store information
   - Provide store name, rating, location, tags, and available products

## Product Operations

1. **Product Search**
   - Use `delivery_product_search_recommand` to search for products based on keywords
   - Consider product names, store names, prices, and tags
   - Present multiple options to users for selection

2. **Product Information**
   - Use `get_delivery_product_info` to retrieve detailed product information
   - Include product name, price, store information, and attributes

## Order Creation

1. **Order Requirements**
   - Only single store orders are supported (all products must be from the same store)
   - Multiple products can be ordered from a single store
   - Required information: user_id, store_id, product_ids, product_cnts, address, dispatch_time

2. **Order Attributes**
   - Collect product attributes (e.g., spice level, portion size) if applicable
   - Record dietary restrictions and special requirements in the order note
   - Do NOT put time requirements in the note field

3. **Dispatch Time**
   - Dispatch time must be in the future
   - Format: yyyy-mm-dd HH:MM:SS
   - Dispatch time is when the rider picks up food from the store

4. **Order Confirmation**
   - After creating an order, confirm all details with the user
   - Ask user to confirm payment before proceeding

## Payment Operations

1. **Payment Process**
   - Use `pay_delivery_order` only after user confirms payment
   - Orders must be in "unpaid" status to be paid
   - Verify order details before payment

2. **Payment Status**
   - Use `get_delivery_order_status` to check current order status
   - Possible statuses: unpaid, paid, cancelled, delivered

## Order Management

1. **Order Modification**
   - Use `modify_delivery_order` to update order notes only
   - Cannot modify products, quantities, or other order details after creation
   - If order is unpaid, user must confirm payment after modification

2. **Order Cancellation**
   - Use `cancel_delivery_order` to cancel orders
   - Cannot cancel orders that are already cancelled
   - Check order status before attempting cancellation

3. **Order Search**
   - Use `search_delivery_orders` to find orders by user_id and status
   - Use `get_delivery_order_detail` to retrieve complete order information

## Delivery Time Calculation

1. **Distance to Time**
   - Use `delivery_distance_to_time` to calculate delivery time from distance
   - Distance is measured in meters
   - Delivery time is returned in minutes

## Best Practices

1. Always verify store and product availability before creating orders
2. Confirm user dietary restrictions and reflect them in order notes
3. Provide clear delivery time estimates to users
4. Verify order status before performing operations (payment, cancellation, modification)
5. Guide users through the complete order flow from search to payment
