# OTA (Online Travel Agency) Domain Policy

## Overview
This policy governs the OTA domain operations including hotel bookings, attraction tickets, flight reservations, and train tickets.

## Hotel Operations

### Hotel Search
- Use `hotel_search_recommand` to search hotels by city and optional keywords
- Consider hotel ratings, star ratings, location, and tags
- Guide users to select a specific hotel before booking

### Hotel Booking
- Use `create_hotel_order` to create hotel bookings
- Required: user_id, hotel_id, room_id, date, quantity
- Date format: YYYY-MM-DD
- Check room availability for the specific date

### Hotel Order Management
- Use `pay_hotel_order` to pay for bookings
- Use `cancel_hotel_order` to cancel bookings
- Use `search_hotel_order` to query user's hotel orders
- Use `get_hotel_order_detail` for detailed order information

## Attraction Operations

### Attraction Search
- Use `attractions_search_recommend` to search attractions by city and keywords
- Consider attraction ratings, descriptions, and locations
- Provide opening hours and ticket price information

### Attraction Booking
- Use `create_attraction_order` to purchase attraction tickets
- Required: user_id, attraction_id, ticket_id, date, quantity
- Date format: YYYY-MM-DD
- Verify ticket availability for the visit date

### Attraction Order Management
- Use `pay_attraction_order` to pay for tickets
- Use `cancel_attraction_order` to cancel tickets
- Use `search_attraction_order` to query user's attraction orders
- Use `get_attraction_order_detail` for detailed order information

## Flight Operations

### Flight Search
- Use `flight_search_recommend` to search flights by departure and arrival cities
- Display flight numbers, departure/arrival times, and airports
- Show available seat types and prices

### Flight Booking
- Use `create_flight_order` to book flight tickets
- Required: user_id, flight_id, seat_id, date, quantity
- Date format: YYYY-MM-DD
- Check seat availability for the flight date

### Flight Order Management
- Use `pay_flight_order` to pay for flight tickets
- Use `modify_flight_order` to change flight date (paid orders only)
- Use `cancel_flight_order` to cancel flight bookings
- Use `search_flight_order` to query user's flight orders
- Use `get_flight_order_detail` for detailed order information

## Train Operations

### Train Search
- Use `train_ticket_search` to search trains by departure city, arrival city, and date
- Display train numbers, departure/arrival times, and stations
- Show available seat types and prices

### Train Booking
- Use `create_train_order` to book train tickets
- Required: user_id, train_id, seat_id, date, quantity
- Date format: YYYY-MM-DD
- Check seat availability for the train date

### Train Order Management
- Use `pay_train_order` to pay for train tickets
- Use `modify_train_order` to change train date (paid orders only)
- Use `cancel_train_order` to cancel train bookings
- Use `search_train_order` to query user's train orders
- Use `get_train_order_detail` for detailed order information

## Order Status Management

### Status Values
- **unpaid**: Order created but not paid
- **paid**: Order has been paid
- **cancelled**: Order has been cancelled

### Status Transitions
- unpaid → paid (via payment)
- unpaid → cancelled (via cancellation)
- paid → cancelled (via cancellation with refund)

## Order Modification Rules

### Flight and Train Orders
- Only paid orders can be modified
- Only the date can be changed
- Price differences are automatically calculated:
  - If new price > old price: order status changes to "unpaid", requires additional payment
  - If new price < old price: difference is refunded
  - If new price = old price: no payment change

### Hotel and Attraction Orders
- Cannot be modified after creation
- Must cancel and create new order if changes needed

## Cancellation and Refund Policy

### Cancellation Rules
- Cannot cancel orders that are already cancelled
- Paid orders: full refund upon cancellation
- Unpaid orders: no refund (no payment made)

### Refund Processing
- Refunds are processed immediately upon cancellation
- Refund amount equals the total price paid

## Search and Query Operations

### Order Search Parameters
- user_id: Required for all searches
- date: Optional filter by order date
- status: Optional filter by order status (default: "paid")

### Information Retrieval
- Use `get_ota_hotel_info` for detailed hotel information
- Use `get_ota_attraction_info` for detailed attraction information
- Use `get_ota_flight_info` for detailed flight information
- Use `get_ota_train_info` for detailed train information

## Best Practices

1. Always validate date format (YYYY-MM-DD) before creating orders
2. Check product availability before booking
3. Confirm order details with user before payment
4. Provide clear information about modification and cancellation policies
5. Guide users through the complete booking flow
6. Handle price differences transparently during modifications
7. Verify order status before attempting modifications or cancellations
