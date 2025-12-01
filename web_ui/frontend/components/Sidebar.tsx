"use client";

import { useState } from "react";
import { Upload, FileText, CheckCircle, AlertCircle } from "lucide-react";
import axios from "axios";

interface SidebarProps {
    onFileSelect: (filename: string) => void;
}

export default function Sidebar({ onFileSelect }: SidebarProps) {
    const [uploading, setUploading] = useState(false);
    const [files, setFiles] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files || e.target.files.length === 0) return;

        const file = e.target.files[0];
        setUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await axios.post("http://localhost:8000/api/upload", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            if (response.data.status === "processed") {
                setFiles((prev) => [...prev, response.data.filename]);
                onFileSelect(response.data.filename);
            }
        } catch (err) {
            setError("Upload failed. Please try again.");
            console.error(err);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="w-64 bg-gray-800 border-r border-gray-700 p-4 flex flex-col h-full">
            <div className="mb-8">
                <h1 className="text-xl font-bold text-blue-400 mb-2">RAG-Anything</h1>
                <p className="text-xs text-gray-400">Multimodal Document Assistant</p>
            </div>

            <div className="mb-6">
                <label
                    htmlFor="file-upload"
                    className={`flex items-center justify-center w-full p-3 rounded-lg border-2 border-dashed border-gray-600 hover:border-blue-500 hover:bg-gray-700 cursor-pointer transition-all ${uploading ? "opacity-50 cursor-not-allowed" : ""
                        }`}
                >
                    <div className="flex flex-col items-center">
                        <Upload className="w-6 h-6 mb-2 text-gray-400" />
                        <span className="text-sm text-gray-300">
                            {uploading ? "Processing..." : "Upload PDF"}
                        </span>
                    </div>
                    <input
                        id="file-upload"
                        type="file"
                        className="hidden"
                        accept=".pdf"
                        onChange={handleUpload}
                        disabled={uploading}
                    />
                </label>
                {error && (
                    <div className="mt-2 text-xs text-red-400 flex items-center">
                        <AlertCircle className="w-3 h-3 mr-1" />
                        {error}
                    </div>
                )}
            </div>

            <div className="flex-1 overflow-y-auto">
                <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                    Processed Files
                </h2>
                <div className="space-y-2">
                    {files.map((file, idx) => (
                        <div
                            key={idx}
                            className="flex items-center p-2 rounded bg-gray-700/50 hover:bg-gray-700 transition-colors cursor-pointer"
                            onClick={() => onFileSelect(file)}
                        >
                            <FileText className="w-4 h-4 text-blue-400 mr-2" />
                            <span className="text-sm text-gray-200 truncate">{file}</span>
                            <CheckCircle className="w-3 h-3 text-green-500 ml-auto" />
                        </div>
                    ))}
                    {files.length === 0 && (
                        <p className="text-xs text-gray-600 italic text-center py-4">
                            No files uploaded yet
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}
