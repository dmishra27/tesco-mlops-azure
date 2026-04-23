"""
Synthetic Tesco transaction event producer for Event Hub.
Generates realistic basket transactions and sends them in batches.
"""

import asyncio
import json
import os
import random
import uuid
from datetime import datetime, timezone

from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData

EVENTHUB_CONNECTION_STRING = os.environ["EVENTHUB_CONNECTION_STRING"]
EVENTHUB_NAME = os.getenv("EVENTHUB_NAME", "transactions")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
INTERVAL_SECONDS = float(os.getenv("INTERVAL_SECONDS", "5"))

CATEGORIES = [
    "ready_meals", "bakery", "produce", "dairy", "meat",
    "beverages", "snacks", "frozen", "household", "health_beauty",
]

STORES = [f"STORE-{i:04d}" for i in range(1, 201)]
CHANNELS = ["in-store", "online"]


def _generate_transaction() -> dict:
    category = random.choice(CATEGORIES)
    quantity = random.randint(1, 8)
    unit_price = round(random.uniform(0.50, 25.00), 2)
    return {
        "transaction_id": str(uuid.uuid4()),
        "customer_id": f"CUST-{random.randint(1, 500_000):07d}",
        "store_id": random.choice(STORES),
        "product_id": f"PROD-{random.randint(1, 50_000):06d}",
        "category": category,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_amount": round(quantity * unit_price, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": random.choice(CHANNELS),
    }


async def send_batch(producer: EventHubProducerClient, batch_size: int) -> int:
    async with producer:
        event_data_batch = await producer.create_batch()
        for _ in range(batch_size):
            payload = json.dumps(_generate_transaction()).encode("utf-8")
            event_data_batch.add(EventData(payload))
        await producer.send_batch(event_data_batch)
    return batch_size


async def main() -> None:
    print(f"Starting producer → Event Hub: {EVENTHUB_NAME}, batch_size={BATCH_SIZE}")
    sent_total = 0
    while True:
        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENTHUB_CONNECTION_STRING,
            eventhub_name=EVENTHUB_NAME,
        )
        n = await send_batch(producer, BATCH_SIZE)
        sent_total += n
        print(f"Sent {n} events (total: {sent_total}) at {datetime.now(timezone.utc).isoformat()}")
        await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
