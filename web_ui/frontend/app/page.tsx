"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
    const [selectedFile, setSelectedFile] = useState<string | null>(null);

    return (
        <main className="flex h-screen bg-gray-900 text-white overflow-hidden">
            <Sidebar onFileSelect={setSelectedFile} />
            <div className="flex-1 flex flex-col">
                <ChatInterface />
            </div>
        </main>
    );
}
