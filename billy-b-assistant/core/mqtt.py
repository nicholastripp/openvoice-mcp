import paho.mqtt.client as mqtt
import json
import os
import subprocess
from core.config import MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
from core.movements import stop_all_motors

mqtt_client = None
mqtt_connected = False

def mqtt_available():
    return all([MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD])

def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        print("🔌 MQTT connected successfully!")
        mqtt_send_discovery()
        client.subscribe("billy/command")
    else:
        print(f"⚠️ MQTT connection failed with code {rc}")

def start_mqtt():
    global mqtt_client
    if not mqtt_available():
        print("⚠️ MQTT not configured, skipping.")
        return

    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        mqtt_client.loop_start()
        mqtt_publish("billy/state", "idle", retain=True)
    except Exception as e:
        print(f"❌ MQTT connection error: {e}")

def stop_mqtt():
    global mqtt_client
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("🔌 MQTT disconnected.")

def mqtt_publish(topic, payload, retain=True, retry=True):
    global mqtt_client, mqtt_connected

    if not mqtt_client or not mqtt_connected:
        if retry:
            print("🔁 MQTT not connected. Trying to reconnect...")
            try:
                mqtt_client.reconnect()
                mqtt_connected = True
            except Exception as e:
                print(f"❌ MQTT reconnect failed: {e}")
                return
        else:
            print(f"⚠️ MQTT not connected. Skipping publish {topic}={payload}")
            return

    try:
        mqtt_client.publish(topic, payload, retain=retain)
        print(f"📡 MQTT publish: {topic} = {payload} (retain={retain})")
    except Exception as e:
        print(f"❌ MQTT publish failed: {e}")

def mqtt_send_discovery():
    """Send MQTT discovery messages for Home Assistant."""
    if not mqtt_client:
        return

    # Sensor for Billy's state
    payload_sensor = {
        "name": "Billy State",
        "unique_id": "billy_state",
        "state_topic": "billy/state",
        "icon": "mdi:fish",
        "device": {
            "identifiers": ["billy_bass"],
            "name": "Big Mouth Billy Bass",
            "model": "Billy Bassistant",
            "manufacturer": "DIY"
        }
    }
    mqtt_client.publish("homeassistant/sensor/billy/state/config", json.dumps(payload_sensor), retain=True)

    # Button to send shutdown command
    payload_button = {
        "name": "Billy Shutdown",
        "unique_id": "billy_shutdown",
        "command_topic": "billy/command",
        "payload_press": "shutdown",
        "device": {
            "identifiers": ["billy_bass"],
            "name": "Big Mouth Billy Bass",
            "model": "Billy Bassistant",
            "manufacturer": "DIY"
        }
    }
    mqtt_client.publish("homeassistant/button/billy/shutdown/config", json.dumps(payload_button), retain=True)

def on_message(client, userdata, msg):
    print(f"📩 MQTT message received: {msg.topic} = {msg.payload.decode()}")
    if msg.topic == "billy/command":
        command = msg.payload.decode().strip().lower()
        if command == "shutdown":
            print("🛑 Shutdown command received over MQTT. Shutting down...")
            try:
                stop_all_motors()
            except Exception as e:
                print(f"⚠️ Error stopping motors: {e}")
            stop_mqtt()
            subprocess.Popen(["sudo", "shutdown", "now"])