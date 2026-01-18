# Instore Domain Policy

## Overview
This policy governs the instore domain operations including shop search, product search, order creation, seat booking, and service reservations.

## Shop Operations

1. **Shop Search**
   - Use `instore_shop_search_recommend` to search for shops based on keywords
   - Consider shop ratings, location, and tags when recommending shops
   - Guide users to select a specific shop before proceeding with orders or bookings

2. **Shop Features**
   - Shops may support seat booking (enable_book flag)
   - Shops may support service reservations (enable_reservation flag)
   - Check shop capabilities before attempting bookings or reservations

## Product Operations

1. **Product Search**
   - Use `instore_product_search_recommend` to search for products based on keywords
   - Consider product names, shop names, prices, and tags
   - Present multiple options to users for selection

2. **Product Orders**
   - Use `create_instore_product_order` to create product orders
   - Required information: user_id, shop_id, product_id, quantity
   - Orders are created in "unpaid" status

## Order Management

1. **Order Creation**
   - Products must belong to the specified shop
   - Quantity must be positive
   - Total price is calculated automatically

2. **Order Payment**
   - Use `pay_instore_order` to pay for orders
   - Orders must be in "unpaid" status to be paid
   - Payment changes status from "unpaid" to "paid"

3. **Order Cancellation**
   - Use `instore_cancel_order` to cancel orders
   - Cannot cancel orders that are already cancelled
   - Cancellation changes status to "cancelled"

4. **Order Retrieval**
   - Use `get_instore_orders` to retrieve all orders for a user
   - Returns list of orders with all details

## Seat Booking Operations

1. **Booking Requirements**
   - Shop must have enable_book=true
   - Required information: user_id, shop_id, time, customer_count
   - Time format: yyyy-mm-dd HH:MM:SS
   - Customer count must be positive

2. **Booking Payment**
   - If book_price is 0, booking is automatically paid
   - If book_price > 0, booking is created in "unpaid" status
   - Use `pay_instore_book` to pay for bookings

3. **Booking Cancellation**
   - Use `instore_cancel_book` to cancel bookings
   - Cannot cancel bookings that are already cancelled

4. **Booking Retrieval**
   - Use `get_instore_books` to retrieve all bookings for a user
   - Use `search_instore_book` to query specific or all bookings

## Service Reservation Operations

1. **Reservation Requirements**
   - Shop must have enable_reservation=true
   - Required information: user_id, shop_id, time, customer_count
   - Time format: yyyy-mm-dd HH:MM:SS
   - Customer count must be positive

2. **Reservation Modification**
   - Use `instore_modify_reservation` to modify reservations
   - Can modify time and/or customer_count
   - Cannot modify reservations with status "consumed" or "cancelled"

3. **Reservation Cancellation**
   - Use `instore_cancel_reservation` to cancel reservations
   - Cannot cancel reservations that are already cancelled

4. **Reservation Retrieval**
   - Use `get_instore_reservations` to retrieve all reservations for a user
   - Use `search_instore_reservation` to query specific or all reservations

## Status Management

### Order Status Values
- **unpaid**: Order created but not paid
- **paid**: Order has been paid
- **cancelled**: Order has been cancelled

### Booking Status Values
- **unpaid**: Booking created but not paid (only if book_price > 0)
- **paid**: Booking has been paid or is free
- **unconsumed**: Booking is paid but service not yet consumed
- **consumed**: Booking service has been consumed
- **cancelled**: Booking has been cancelled

### Reservation Status Values
- **unpaid**: Reservation created but not paid
- **paid**: Reservation has been paid
- **unconsumed**: Reservation is paid but service not yet consumed
- **consumed**: Reservation service has been consumed
- **cancelled**: Reservation has been cancelled

## Best Practices

1. Always verify shop capabilities before creating bookings or reservations
2. Validate time format before submitting bookings or reservations
3. Check status before attempting payment or cancellation operations
4. Provide clear information about booking fees to users
5. Guide users through the complete flow from search to payment
6. Handle modification requests appropriately based on current status
