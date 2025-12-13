import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-routing-machine';
import { MapPin, Navigation, Car } from 'lucide-react';
import { renderToString } from 'react-dom/server';

// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom marker icons
const createCustomIcon = (IconComponent, color) => {
    const iconHtml = renderToString(
        <div style={{
            background: color,
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            border: '3px solid white'
        }}>
            <IconComponent size={20} color="white" />
        </div>
    );

    return L.divIcon({
        html: iconHtml,
        className: 'custom-marker',
        iconSize: [40, 40],
        iconAnchor: [20, 20],
    });
};

const pickupIcon = createCustomIcon(MapPin, '#00D277');
const dropIcon = createCustomIcon(Navigation, '#FF4757');
const carIcon = createCustomIcon(Car, '#000000');

// Component to handle routing
function RoutingMachine({ pickup, drop }) {
    const map = useMap();
    const routingControlRef = useRef(null);

    useEffect(() => {
        if (!map || !pickup || !drop) return;

        // Remove existing routing control
        if (routingControlRef.current) {
            map.removeControl(routingControlRef.current);
        }

        // Create new routing control
        const routingControl = L.Routing.control({
            waypoints: [
                L.latLng(pickup.lat, pickup.lng),
                L.latLng(drop.lat, drop.lng)
            ],
            routeWhileDragging: false,
            addWaypoints: false,
            draggableWaypoints: false,
            fitSelectedRoutes: true,
            showAlternatives: false,
            lineOptions: {
                styles: [{ color: '#00D277', weight: 6, opacity: 0.8 }]
            },
            createMarker: () => null, // We use custom markers
        }).addTo(map);

        routingControlRef.current = routingControl;

        // Hide the routing instructions panel
        const container = routingControl.getContainer();
        if (container) {
            container.style.display = 'none';
        }

        return () => {
            if (routingControlRef.current) {
                map.removeControl(routingControlRef.current);
            }
        };
    }, [map, pickup, drop]);

    return null;
}

// Component to fit bounds when markers change
function FitBounds({ pickup, drop }) {
    const map = useMap();

    useEffect(() => {
        if (pickup && drop) {
            const bounds = L.latLngBounds([
                [pickup.lat, pickup.lng],
                [drop.lat, drop.lng]
            ]);
            map.fitBounds(bounds, { padding: [50, 50] });
        } else if (pickup) {
            map.setView([pickup.lat, pickup.lng], 13);
        }
    }, [map, pickup, drop]);

    return null;
}

export default function Map({ pickup, drop, nearbyCars = [], onMapClick }) {
    const defaultCenter = [12.9716, 77.5946]; // Bangalore
    const defaultZoom = 13;

    return (
        <div className="h-full w-full relative">
            <MapContainer
                center={pickup ? [pickup.lat, pickup.lng] : defaultCenter}
                zoom={defaultZoom}
                className="h-full w-full z-0"
                zoomControl={true}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Pickup Marker */}
                {pickup && (
                    <Marker
                        position={[pickup.lat, pickup.lng]}
                        icon={pickupIcon}
                        draggable={true}
                        eventHandlers={{
                            dragend: (e) => {
                                const { lat, lng } = e.target.getLatLng();
                                onMapClick?.({ type: 'pickup', lat, lng });
                            }
                        }}
                    >
                        <Popup>
                            <div className="text-sm font-semibold">
                                📍 Pickup Location
                                <br />
                                <span className="text-xs text-gray-600">{pickup.address || 'Drag to adjust'}</span>
                            </div>
                        </Popup>
                    </Marker>
                )}

                {/* Drop Marker */}
                {drop && (
                    <Marker
                        position={[drop.lat, drop.lng]}
                        icon={dropIcon}
                        draggable={true}
                        eventHandlers={{
                            dragend: (e) => {
                                const { lat, lng } = e.target.getLatLng();
                                onMapClick?.({ type: 'drop', lat, lng });
                            }
                        }}
                    >
                        <Popup>
                            <div className="text-sm font-semibold">
                                🎯 Drop Location
                                <br />
                                <span className="text-xs text-gray-600">{drop.address || 'Drag to adjust'}</span>
                            </div>
                        </Popup>
                    </Marker>
                )}

                {/* Nearby Cars */}
                {nearbyCars.map((car, idx) => (
                    <Marker
                        key={idx}
                        position={[car.lat, car.lng]}
                        icon={carIcon}
                    >
                        <Popup>
                            <div className="text-sm">
                                🚗 {car.type || 'Available Car'}
                                <br />
                                <span className="text-xs text-gray-600">{car.eta || '2 mins'} away</span>
                            </div>
                        </Popup>
                    </Marker>
                ))}

                {/* Routing */}
                {pickup && drop && <RoutingMachine pickup={pickup} drop={drop} />}

                {/* Auto-fit bounds */}
                <FitBounds pickup={pickup} drop={drop} />
            </MapContainer>
        </div>
    );
}
