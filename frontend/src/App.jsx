import { useState } from 'react'
import axios from 'axios'
import Map from './components/Map'
import BottomSheet from './components/BottomSheet'
import VehicleCard from './components/VehicleCard'
import BookingModal from './components/BookingModal'
import { TrendingUp } from 'lucide-react'
import './App.css'

function App() {
  // State for map locations
  const [pickup, setPickup] = useState(null)
  const [drop, setDrop] = useState(null)

  // State for API
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  // State for booking
  const [selectedVehicle, setSelectedVehicle] = useState(null)
  const [showBookingModal, setShowBookingModal] = useState(false)

  // Simulated nearby cars
  const [nearbyCars] = useState([
    { lat: 12.9756, lng: 77.5986, type: 'Standard', eta: '2 mins' },
    { lat: 12.9686, lng: 77.6006, type: 'Premium', eta: '5 mins' },
    { lat: 12.9726, lng: 77.5916, type: 'Eco', eta: '3 mins' },
  ])

  // Handle search - locations come from autocomplete
  const handleSearch = async () => {
    setLoading(true)
    setError('')
    setResult(null)

    try {
      // Check if we have both locations selected
      if (!pickup || !drop) {
        setError('Please select both pickup and drop locations from the suggestions')
        setLoading(false)
        return
      }

      // Call AI Backend with coordinates from autocomplete
      const payload = {
        origin_lat: pickup.lat,
        origin_lon: pickup.lng,
        dest_lat: drop.lat,
        dest_lon: drop.lng,
        user_preference: "balanced"
      }

      const backendResponse = await axios.post('http://127.0.0.1:8000/ride/quote', payload)
      setResult(backendResponse.data)

    } catch (err) {
      setError(err.message || "Something went wrong connecting to the AI.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Handle booking
  const handleBookVehicle = (vehicle) => {
    setSelectedVehicle(vehicle)
    setShowBookingModal(true)
  }

  const handleCloseModal = () => {
    setShowBookingModal(false)
    // Reset after animation
    setTimeout(() => {
      setSelectedVehicle(null)
    }, 300)
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-gray-100 relative">
      {/* Map */}
      <Map
        pickup={pickup}
        drop={drop}
        nearbyCars={nearbyCars}
        onMapClick={(data) => {
          if (data.type === 'pickup') {
            setPickup({ lat: data.lat, lng: data.lng, address: 'Adjusted pickup' })
          } else if (data.type === 'drop') {
            setDrop({ lat: data.lat, lng: data.lng, address: 'Adjusted drop' })
          }
        }}
      />

      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-white/90 backdrop-blur-md shadow-sm">
        <div className="flex items-center justify-between px-5 py-4">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-ola-primary rounded-full flex items-center justify-center text-white font-bold text-lg">
              🚖
            </div>
            <div>
              <h1 className="text-lg font-black text-gray-800">AI Ride</h1>
              <p className="text-xs text-gray-600">Powered by ML</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center">
              👤
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Sheet */}
      <BottomSheet
        pickup={pickup}
        drop={drop}
        onPickupChange={setPickup}
        onDropChange={setDrop}
        onSearch={handleSearch}
        loading={loading}
      >
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg mb-4">
            <p className="text-red-700 text-sm font-medium">{error}</p>
          </div>
        )}

        {result && (
          <div className="space-y-4">
            {/* Stats Row */}
            <div className="grid grid-cols-3 gap-3 bg-gradient-to-br from-ola-primary/10 to-green-50 p-4 rounded-2xl">
              <div className="text-center">
                <p className="text-xs text-gray-600 mb-1">Distance</p>
                <p className="text-lg font-bold text-gray-800">{result.trip_distance_km} km</p>
              </div>
              <div className="text-center border-l border-r border-gray-300">
                <p className="text-xs text-gray-600 mb-1">Demand</p>
                <p className="text-lg font-bold text-gray-800">{result.predicted_demand_next_hour}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-600 mb-1">Surge</p>
                <p className={`text-lg font-bold ${result.surge_applied !== '1.0x' ? 'text-red-500' : 'text-ola-primary'
                  }`}>
                  {result.surge_applied}
                </p>
              </div>
            </div>

            {/* AI Prediction Badge */}
            <div className="flex items-center gap-2 bg-gradient-to-r from-purple-100 to-blue-100 p-3 rounded-xl">
              <TrendingUp size={18} className="text-purple-600" />
              <p className="text-sm text-purple-900 font-medium">
                AI optimized pricing • LSTM + XGBoost models
              </p>
            </div>

            {/* Vehicle Cards */}
            <div className="space-y-3">
              <h3 className="font-bold text-gray-800">Available Rides</h3>
              {result.options.map((vehicle, idx) => (
                <VehicleCard
                  key={idx}
                  vehicle={vehicle}
                  index={idx}
                  onBook={handleBookVehicle}
                  isRecommended={idx === 0}
                />
              ))}
            </div>
          </div>
        )}
      </BottomSheet>

      {/* Booking Modal */}
      <BookingModal
        isOpen={showBookingModal}
        onClose={handleCloseModal}
        vehicle={selectedVehicle}
        pickup={pickup}
        drop={drop}
        distance={result?.trip_distance_km}
      />
    </div>
  )
}

export default App