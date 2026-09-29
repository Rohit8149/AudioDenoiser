import json
import threading
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
BASE_TOPIC = "aml_lab_2026/audiodenoiser"

class RemoteController:
    def __init__(self, device_id, on_command_callback, on_status_callback):
        self.device_id = device_id
        self.on_command = on_command_callback
        self.on_status = on_status_callback
        
        self.cmd_topic = f"{BASE_TOPIC}/command/{self.device_id}"
        self.state_topic = f"{BASE_TOPIC}/state/{self.device_id}"
        
        self.client = mqtt.Client(client_id=f"aml_client_{self.device_id}_{threading.get_ident()}")
        
        # Last Will and Testament (LWT)
        # If this PC crashes, the broker will automatically broadcast this "offline" message.
        will_payload = json.dumps({"device_id": self.device_id, "status": "offline"})
        self.client.will_set(self.state_topic, payload=will_payload, qos=1, retain=True)
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
    def start(self):
        self.on_status("Connecting to Cloud...", False)
        try:
            self.client.connect(BROKER, PORT, 60)
            self.client.loop_start()
        except Exception as e:
            self.on_status(f"Network Error: {e}", True)

    def stop(self):
        # Graceful disconnect: manually publish the offline state, then stop.
        offline_payload = json.dumps({"device_id": self.device_id, "status": "offline"})
        self.client.publish(self.state_topic, payload=offline_payload, qos=1, retain=True)
        
        self.client.loop_stop()
        self.client.disconnect()

    def publish_state(self, role, denoise_on, isolate_on, cocktail_on, gain):
        """Broadcasts the current state of the app to the cloud using Retained Messages."""
        payload = json.dumps({
            "device_id": self.device_id,
            "role": role,
            "status": "online",
            "denoise": denoise_on,
            "isolate": isolate_on,
            "cocktail": cocktail_on,
            "gain": gain
        })
        # retain=True tells the cloud to save this message for late-joiner Admins!
        self.client.publish(self.state_topic, payload=payload, qos=1, retain=True)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.on_status(f"Connected (ID: {self.device_id})", False)
            self.client.subscribe(self.cmd_topic)
        else:
            self.on_status(f"Failed to connect (Code {rc})", True)

    def _on_disconnect(self, client, userdata, rc):
        self.on_status("Disconnected from Cloud.", True)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            action = payload.get("action")
            state = payload.get("state")
            
            if action is not None and state is not None:
                self.on_command(action, state)
        except Exception as e:
            print(f"MQTT Parse Error: {e}")
