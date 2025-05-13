db.createCollection("logs", {
    "capped": false,
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "title": "logs",
            "properties": {
                "_id": {
                    "bsonType": "objectId"
                },
                "event_id": {
                    "bsonType": "objectId"
                },
                "timestamp": {
                    "bsonType": "string"
                },
                "level": {
                    "bsonType": "string"
                },
                "event": {
                    "bsonType": "string"
                },
                "message": {
                    "bsonType": "string"
                },
                "device_info": {
                    "bsonType": "object",
                    "properties": {
                        "os": {
                            "bsonType": "string"
                        },
                        "ipv4": {
                            "bsonType": "string"
                        }
                    },
                    "additionalProperties": false
                }
            },
            "additionalProperties": false,
            "required": [
                "event"
            ]
        }
    },
    "validationLevel": "off",
    "validationAction": "warn"
});



db.createCollection("session_data", {
    "capped": false,
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "title": "session_data",
            "properties": {
                "_id": {
                    "bsonType": "objectId"
                },
                "session_id": {
                    "bsonType": "objectId"
                },
                "customer_id": {
                    "bsonType": "objectId"
                },
                "start_time": {
                    "bsonType": "string"
                },
                "stop_time": {
                    "bsonType": "string"
                },
                "events": {
                    "bsonType": "array",
                    "additionalItems": true,
                    "items": {
                        "bsonType": "object",
                        "properties": {
                            "type": {
                                "bsonType": "string"
                            },
                            "target": {
                                "bsonType": "string"
                            },
                            "timestamp": {
                                "bsonType": "string"
                            }
                        },
                        "additionalProperties": false
                    }
                },
                "device_info": {
                    "bsonType": "object",
                    "properties": {
                        "os": {
                            "bsonType": "string"
                        },
                        "browser": {
                            "bsonType": "string"
                        },
                        "ipv4": {
                            "bsonType": "string"
                        }
                    },
                    "additionalProperties": false
                }
            },
            "additionalProperties": false
        }
    },
    "validationLevel": "off",
    "validationAction": "warn"
});



db.createCollection("customer_feedback", {
    "capped": false,
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "title": "customer_feedback",
            "properties": {
                "_id": {
                    "bsonType": "objectId"
                },
                "fdbck_d": {
                    "bsonType": "objectId",
                    "title": "feedback_id"
                },
                "customer_id": {
                    "bsonType": "objectId"
                },
                "channel": {
                    "bsonType": "string"
                },
                "timestamp": {
                    "bsonType": "string"
                },
                "message": {
                    "bsonType": "string"
                }
            },
            "additionalProperties": false
        }
    },
    "validationLevel": "off",
    "validationAction": "warn"
});