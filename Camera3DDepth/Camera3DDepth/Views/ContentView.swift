//
//  ContentView.swift
//  Camera3DDepth
//
//  Created by Giorgio Mancusi on 9/27/24.
//

import SwiftUI

struct ContentView: View {
        
    @State private var ipAddress: String = ""
    @State private var isOnline: Bool = false
    @State private var showJet: Bool = false
    @State private var jetImage: UIImage?
    
    var body: some View {
        VStack {
            VStack {
                Toggle("Enable Online Mode", isOn: $isOnline)
                TextField("IP Address", text: $ipAddress)
                
                Toggle("Show JET Image", isOn: $showJet)

            }
            .padding()

            ARViewContainer(showJetColorPreview: $showJet, jetImage: $jetImage)
                .edgesIgnoringSafeArea(.all)
                .frame(height: 400)

            
            if showJet, let jetImage = jetImage {
                Image(uiImage: jetImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 300, height: 300)
                    .padding()
            }
            else {
                Text("Jet View is not available.")
                    .padding()
            }
        }
    }
}

