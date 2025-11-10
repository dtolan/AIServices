import { useState, useRef } from 'react'
import { FiUpload, FiX, FiImage } from 'react-icons/fi'

export default function ImageUpload({ onImageSelect, selectedImage, onClear }) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file')
      return
    }

    // Convert to base64
    const reader = new FileReader()
    reader.onloadend = () => {
      const base64String = reader.result.split(',')[1] // Remove data:image/...;base64, prefix
      onImageSelect(base64String, file.name)
    }
    reader.readAsDataURL(file)
  }

  const handleButtonClick = () => {
    inputRef.current?.click()
  }

  const handleClear = () => {
    if (inputRef.current) {
      inputRef.current.value = ''
    }
    onClear()
  }

  return (
    <div className="w-full">
      {!selectedImage ? (
        <div
          className={`relative border-2 border-dashed rounded-lg p-6 transition-all ${
            dragActive
              ? 'border-primary-500 bg-primary-500/10'
              : 'border-white/20 hover:border-primary-500/50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept="image/*"
            onChange={handleChange}
          />

          <div className="flex flex-col items-center justify-center text-center">
            <FiUpload className="w-12 h-12 mb-3 text-white/50" />
            <p className="text-white/70 mb-2">Drag and drop an image here, or</p>
            <button
              onClick={handleButtonClick}
              className="btn-primary flex items-center space-x-2"
            >
              <FiImage className="w-4 h-4" />
              <span>Browse Files</span>
            </button>
            <p className="text-sm text-white/50 mt-3">
              Supports: JPG, PNG, WebP, and more
            </p>
          </div>
        </div>
      ) : (
        <div className="relative group">
          <img
            src={`data:image/png;base64,${selectedImage}`}
            alt="Selected source"
            className="w-full rounded-lg"
          />
          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
            <button
              onClick={handleClear}
              className="btn-secondary flex items-center space-x-2"
            >
              <FiX className="w-5 h-5" />
              <span>Remove Image</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
