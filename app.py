"""
Aerial Imagery Analysis Dashboard — Terra Drone App.
Includes multi-language support, Single-Image view, and Comparison Mode 
with a high-performance Weather Radar style Hotspot Heatmap, dynamic context-aware color legends,
and an automated ecosystem mitigation planner with refined typography and clean layout.
"""

import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Language Configuration
LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "zh": "中文",
}

TRANSLATIONS = {
    "en": {
        "app_title": "Aerial Imagery Analysis",
        "app_description": "Upload drone or satellite imagery to analyze vegetation, water, soil, sand, and other surfaces.",
        "upload_imagery": "Upload aerial imagery",
        "supported_formats": "Supported formats: JPEG, PNG, GeoTIFF",
        "imagery_view": "Imagery View",
        "select_view_mode": "Select view mode:",
        "original_image": "Original Image",
        "segmented_view": "Segmented View",
        "segmented_view_caption": "Segmented View",
        "resolution": "Resolution",
        "segmented_view_not_available": "Segmented view not available or image not yet processed.",
        "uploaded_imagery_title": "Uploaded Imagery",
        "upload_to_preview": "Upload an image to preview it here.",
        "area_breakdown": "Area Breakdown Matrix",
        "analysis_results": "Analysis Results",
        "vegetation": "Vegetation Canopy Cover",
        "water": "Hydrological Surface Bodies",
        "other": "Urban Sprawl & Built Infrastructure",
        "soil": "Exposed Soil Matrix",
        "sand": "Desertification & Arid Land",
        "upload_to_run_analysis": "Upload an image to run analysis.",
        "images_analyzed": "Distinct Images Processed",
        "last_processed": "Last processed",
        "terra_drone_title": "Terra Drone",
        "dashboard_caption": "Aerial Imagery Analysis Dashboard",
        "processing_spinner": "Analyzing image layers...",
        "analysis_complete": "Image analysis complete.",
        "density_grids": "Density Grids (16x16)",
        "vegetation_density": "Vegetation Density (%)",
        "water_density": "Water Density (%)",
        "other_density": "Urban Sprawl Density (%)",
        "soil_density": "Soil Density (%)",
        "sand_density": "Arid Land Density (%)",
        "select_language": "Select your language:",
        "welcome_message": "Welcome to Terra Drone! Please select your preferred language to continue.",
        "session": "Session Stats",
        "select_mode_title": "Select Operation Mode",
        "select_mode_desc": "Choose how you would like to process your aerial assets today:",
        "mode_single": "Single Image Analysis",
        "mode_compare": "Image Comparison Mode",
        "upload_img_1": "Upload Baseline Image (Image 1)",
        "upload_img_2": "Upload Comparison Image (Image 2)",
        "comparison_dashboard": "Change Detection & Comparison Matrix",
        "metric_label": "Feature Class",
        "img1_label": "Image 1 (%)",
        "img2_label": "Image 2 (%)",
        "diff_label": "Difference (Δ %)",
        # Newly Localized UI Elements
        "change_mode": "↩️ Change operation mode",
        "full_app_reset": "🔄 Full App Reset",
        "start_new_comparison": "🔄 Start New Comparison",
        "hotspot_settings": "🔥 Hotspot Heatmap Settings",
        "select_hotspot_layer": "Select Hotspot Layer:",
        "overlay_intensity": "Overlay Intensity",
        "image_1_baseline": "📷 Image 1 (Baseline)",
        "image_2_comparison": "📷 Image 2 (Comparison)",
        "view_raw_matrix": "📊 View Raw Matrix Density Grids",
        "show_mitigation": "📋 Show Mitigation Suggestions",
        "mitigation_planner_title": "📋 Environmental Intervention & Mitigation Planner",
        "biomass_loss": "Biomass Loss Detected",
        "biomass_growth": "Biomass Growth",
        "hydrological_loss": "Hydrological Loss",
        "stable_volume": "Stable Volume",
        "critical_dryout": "🚨 CRITICAL DRY OUT EVENT",
        "controlled_dev": "■ Controlled Development Threshold",
        "expansion_spike": "Expansion Spike Detected",
        "topsoil_exposure": "▲ Topsoil Exposure Increase",
        "stabilized_surface": "Stabilized Surface Cover",
        "card_veg": "Amount of Vegetation",
        "card_water": "Water Scarcity Indicator",
        "card_sprawl": "Infrastructure Expansion",
        "card_soil": "Exposed Soil Matrix",
        "hotspot_none": "None",
        "hotspot_urban": "Urban Sprawl Trends",
        "hotspot_loss": "Canopy Loss Matrix",
        "hotspot_healthy": "Healthy Canopy Tracker",
        # SDG Localization
        "sdg_intro": "Welcome to the <strong>Terra Drone Imagery Suite</strong>, an advanced intelligence application engineered to transform high-resolution drone, aerial, and satellite imagery into spatial, actionable data layers. By combining automatic computer vision classification masks with high-fidelity thermal grid structures, this suite provides direct monitoring metrics for environmental impact assessment, ecological change tracking, and master infrastructure planning.",
        "sdg_title": "🌿 Alignment with UN Sustainable Development Goal 11",
        "sdg_desc": "<strong>SDG 11: Sustainable Cities and Communities</strong> aims to make human settlements inclusive, safe, resilient, and sustainable. This application actively drives this mission through targeted environmental and geo-spatial features:",
        "sdg_target_1": "<strong>Target 11.3 (Inclusive & Sustainable Urbanization):</strong> By monitoring the <strong>Urban Sprawl & Built Infrastructure</strong> indices, urban planners can monitor structural boundaries over time to prevent uncontrolled horizontal sprawl and protect natural zones.",
        "sdg_target_2": "<strong>Target 11.7 (Provide Access to Green & Public Spaces):</strong> The high-performance <strong>Vegetation Canopy Cover Matrix</strong> provides developers with concrete coverage metrics to preserve, map, and expand green spaces, mitigating the urban heat island effect.",
        "sdg_target_3": "<strong>Target 11.B (Disaster Risk Reduction & Environmental Resilience):</strong> By utilizing the <strong>Image Comparison Change Detection Engine</strong>, regions can actively evaluate environmental transformations, desertification dynamics, and surface water vulnerabilities to adjust ecological mitigation strategies.",
        "btn_select_single": "Select Single Mode",
        "btn_select_compare": "Select Comparison Mode",
        # Mitigation Text
        "mit_veg_low_title": "**🚨 Low Biomass Profile Triggered**",
        "mit_veg_low_desc": "* Launch targeted re-seeding operations or agroforestry barriers.\n* Restrict structural clearing in vulnerable quadrants.",
        "mit_veg_mod_title": "**⚠️ Moderate Canopy Biomass Distribution**",
        "mit_veg_mod_desc": "* Localized field maintenance requested.\n* Limit timber permits in buffer territories.",
        "mit_veg_opt_title": "**✅ Optimal Biomass Density Core Verified**",
        "mit_veg_opt_desc": "* Normal natural stability indices confirmed.",
        "mit_wat_low_title": "**⚠️ Hydrological Scarcity Profile Alert**",
        "mit_wat_low_desc": "* Deploy evaporation barriers or runoff restrictions over surface pools.",
        "mit_wat_opt_title": "**💧 Water Buffer Within Standard Ranges**",
        "mit_wat_opt_desc": "* Standard runtime monitoring remains sufficient.",
        "mit_urb_high_title": "**🏗️ High Infrastructure Expansion Spotted**",
        "mit_urb_high_desc": "* Enforce permeable materials configurations for new builds.\n* Integrate urban micro-forest corridors to mitigate thermal effects.",
        "mit_urb_opt_title": "**🌲 Balanced Urban-to-Nature Asset Matrix**",
        "mit_urb_opt_desc": "* Zoning bounds healthy.",
        "mit_arid_high_title": "**🏜️ Active Desertification / Soil Exposure Signal**",
        "mit_arid_high_desc": "* Establish deep root brush patterns across sandy bounds.\n* Install perpendicular windbreak borders.",
        # Legends
        "leg_loss_title": "📉 Canopy Loss Tracker Legend",
        "leg_loss_sub": "Targeting regions experiencing tree removal events",
        "leg_loss_1": "🟢 Intact", "leg_loss_2": "🟡 Minor", "leg_loss_3": "🟠 Moderate", "leg_loss_4": "🔴 High", "leg_loss_5": "🟣 Critical",
        "leg_health_title": "🌿 Healthy Canopy Tracker Legend",
        "leg_health_sub": "Mapping vegetation biomass and structural density",
        "leg_health_1": "🟢 Sparse", "leg_health_2": "🟡 Ground", "leg_health_3": "🟠 Medium", "leg_health_4": "🔴 Dense", "leg_health_5": "🟣 Peak",
        "leg_sprawl_title": "🏗️ Urban Sprawl Trends Legend",
        "leg_sprawl_sub": "Identifying structural growth and artificial paving",
        "leg_sprawl_1": "🟢 Rural", "leg_sprawl_2": "🟡 Low Built", "leg_sprawl_3": "🟠 Developing", "leg_sprawl_4": "🔴 High Built", "leg_sprawl_5": "🟣 Paved",
        "leg_gen_title": "🔥 Heatmap Intensity Legend",
        "leg_gen_sub": "Weather Spectrum Mapping Layout",
        "leg_gen_1": "🟢 Low", "leg_gen_2": "🟡 Moderate", "leg_gen_3": "🟠 High", "leg_gen_4": "🔴 Severe", "leg_gen_5": "🟣 Peak"
    },
    "es": {
        "app_title": "Análisis de Imágenes Aéreas",
        "app_description": "Suba imágenes de drones o satélites para analizar la vegetación, el agua, el suelo, la arena y otras superficies.",
        "upload_imagery": "Subir imágenes aéreas",
        "supported_formats": "Formatos soportados: JPEG, PNG, GeoTIFF",
        "imagery_view": "Vista de Imágenes",
        "select_view_mode": "Seleccione el modo de vista:",
        "original_image": "Imagen Original",
        "segmented_view": "Vista Segmentada",
        "segmented_view_caption": "Vista Segmentada",
        "resolution": "Resolución",
        "segmented_view_not_available": "Vista segmentada no disponible o imagen aún no procesada.",
        "uploaded_imagery_title": "Imágenes Subidas",
        "upload_to_preview": "Suba una imagen para ver la vista previa aquí.",
        "area_breakdown": "Matriz de Desglose de Área",
        "analysis_results": "Resultados del Análisis",
        "vegetation": "Cobertura del Dosel de Vegetación",
        "water": "Cuerpos de Superficie Hidrológica",
        "other": "Expansión Urbana e Infraestructura Construida",
        "soil": "Matriz de Suelo Expuesto",
        "sand": "Desertificación y Tierra Árida",
        "upload_to_run_analysis": "Suba una imagen para ejecutar el análisis.",
        "images_analyzed": "Imágenes Distintas Procesadas",
        "last_processed": "Última procesada",
        "terra_drone_title": "Terra Drone",
        "dashboard_caption": "Panel de Análisis de Imágenes Aéreas",
        "processing_spinner": "Analizando capas de imagen...",
        "analysis_complete": "Análisis de imagen completo.",
        "density_grids": "Cuadrículas de Densidad (16x16)",
        "vegetation_density": "Densidad de Vegetación (%)",
        "water_density": "Densidad de Agua (%)",
        "other_density": "Densidad de Expansión Urbana (%)",
        "soil_density": "Densidad de Suelo (%)",
        "sand_density": "Densidad de Tierra Árida (%)",
        "select_language": "Seleccione su idioma:",
        "welcome_message": "¡Bienvenido a Terra Drone! Por favor seleccione su idioma de preferencia para continuar.",
        "session": "Estadísticas de la Sesión",
        "select_mode_title": "Seleccionar Modo de Operación",
        "select_mode_desc": "Elija cómo desea procesar sus activos aéreos hoy:",
        "mode_single": "Análisis de Imagen Única",
        "mode_compare": "Modo de Comparación de Imágenes",
        "upload_img_1": "Subir Imagen de Referencia (Imagen 1)",
        "upload_img_2": "Subir Imagen de Comparación (Imagen 2)",
        "comparison_dashboard": "Matriz de Comparación y Detección de Cambios",
        "metric_label": "Clase de Característica",
        "img1_label": "Imagen 1 (%)",
        "img2_label": "Imagen 2 (%)",
        "diff_label": "Diferencia (Δ %)",
        # Newly Localized UI Elements
        "change_mode": "↩️ Cambiar modo de operación",
        "full_app_reset": "🔄 Reinicio completo de la aplicación",
        "start_new_comparison": "🔄 Iniciar nueva comparación",
        "hotspot_settings": "🔥 Configuración del mapa de calor",
        "select_hotspot_layer": "Seleccione capa de puntos críticos:",
        "overlay_intensity": "Intensidad de superposición",
        "image_1_baseline": "📷 Imagen 1 (Base)",
        "image_2_comparison": "📷 Imagen 2 (Comparación)",
        "view_raw_matrix": "📊 Ver matrices de densidad de cuadrícula",
        "show_mitigation": "📋 Mostrar sugerencias de mitigación",
        "mitigation_planner_title": "📋 Planificador de Mitigación e Intervención Ambiental",
        "biomass_loss": "Pérdida de biomasa detectada",
        "biomass_growth": "Crecimiento de biomasa",
        "hydrological_loss": "Pérdida hidrológica",
        "stable_volume": "Volumen estable",
        "critical_dryout": "🚨 EVENTO CRÍTICO DE SEQUÍA",
        "controlled_dev": "■ Umbral de desarrollo controlado",
        "expansion_spike": "Pico de expansión detectado",
        "topsoil_exposure": "▲ Aumento de exposición de capa superficial",
        "stabilized_surface": "Cobertura de superficie estabilizada",
        "card_veg": "Cantidad de Vegetación",
        "card_water": "Indicador de Escasez de Agua",
        "card_sprawl": "Expansión de Infraestructura",
        "card_soil": "Matriz de Suelo Expuesto",
        "hotspot_none": "Ninguno",
        "hotspot_urban": "Tendencias de Expansión Urbana",
        "hotspot_loss": "Matriz de Pérdida de Dosel",
        "hotspot_healthy": "Rastreador de Dosel Saludable",
        # SDG Localization
        "sdg_intro": "Bienvenido a <strong>Terra Drone Imagery Suite</strong>, una aplicación de intelligence avanzada diseñada para transformar imágenes satelitales, aéreas y de drones de alta resolución en capas de datos espaciales y procesables. Al combinar máscaras automáticas de clasificación por visión informática con estructuras de cuadrícula térmica de alta fidelidad, esta suite proporciona métricas de monitoreo directo para la evaluación del impacto ambiental, el seguimiento de cambios ecológicos y la planificación de infraestructura.",
        "sdg_title": "🌿 Alineación con el Objetivo de Desarrollo Sostenible 11 de la ONU",
        "sdg_desc": "El <strong>ODS 11: Ciudades y Comunidades Sostenibles</strong> tiene como objetivo lograr que los asentamientos humanos sean inclusivos, seguros, resilientes y sostenibles. Esta aplicación impulsa activamente esta misión a través de características ambientales y geoespaciales específicas:",
        "sdg_target_1": "<strong>Meta 11.3 (Urbanización inclusiva y sostenible):</strong> Al monitorear los índices de <strong>Expansión Urbana e Infraestructura Construida</strong>, los planificadores urbanos pueden controlar los límites estructurales a lo largo del tiempo para evitar la expansión horizontal descontrolada y proteger las zonas naturales.",
        "sdg_target_2": "<strong>Meta 11.7 (Proporcionar acceso a espacios verdes y públicos):</strong> La <strong>Matriz de Cobertura del Dosel de Vegetación</strong> de alto rendimiento proporciona a los desarrolladores métricas de cobertura concretas para preservar, mapear y expandir los espacios verdes, mitigando el efecto de isla de calor urbana.",
        "sdg_target_3": "<strong>Meta 11.B (Reducción del riesgo de desastres y resiliencia ambiental):</strong> Al utilizar el <strong>Motor de Detección de Cambios de Comparación de Imágenes</strong>, las regiones pueden evaluar activamente las transformaciones ambientales, la dinámica de desertificación y las vulnerabilidades del agua superficial para ajustar las estrategias de mitigación ecológica.",
        "btn_select_single": "Seleccionar modo único",
        "btn_select_compare": "Seleccionar modo de comparación",
        # Mitigation Text
        "mit_veg_low_title": "**🚨 Perfil de Baja Biomasa Activado**",
        "mit_veg_low_desc": "* Iniciar operaciones de siembra dirigida o barreras agroforestales.\n* Restringir la limpieza estructural en cuadrantes vulnerables.",
        "mit_veg_mod_title": "**⚠️ Distribución Moderada de Biomasa del Dosel**",
        "mit_veg_mod_desc": "* Se solicita mantenimiento de campo localizado.\n* Limitar los permisos de tala en territorios de amortiguamiento.",
        "mit_veg_opt_title": "**✅ Núcleo de Densidad de Biomasa Óptimo Verificado**",
        "mit_veg_opt_desc": "* Se confirman índices normales de estabilidad natural.",
        "mit_wat_low_title": "**⚠️ Alerta de Perfil de Escasez Hidrológica**",
        "mit_wat_low_desc": "* Desplegar barreras de evaporación o restricciones de escorrentía sobre los estanques superficiales.",
        "mit_wat_opt_title": "**💧 Amortiguador de Agua Dentro de Rangos Estándar**",
        "mit_wat_opt_desc": "* El monitoreo estándar en tiempo de ejecución sigue siendo suficiente.",
        "mit_urb_high_title": "**🏗️ Alta Expansión de Infraestructura Detectada**",
        "mit_urb_high_desc": "* Aplicar configuraciones de materiales permeables para nuevas construcciones.\n* Integrar corredores de microclima urbano para mitigar efectos térmicos.",
        "mit_urb_opt_title": "**🌲 Matriz de Activos Equilibrada entre lo Urbano y lo Natural**",
        "mit_urb_opt_desc": "* Los límites de zonificación son saludables.",
        "mit_arid_high_title": "**🏜️ Señal Activa de Desertificación / Exposición del Suelo**",
        "mit_arid_high_desc": "* Establecer patrones de maleza de raíz profunda en límites arenosos.\n* Instalar barreras rompevientos perpendiculares.",
        # Legends
        "leg_loss_title": "📉 Leyenda del Rastreador de Pérdida de Dosel",
        "leg_loss_sub": "Dirigido a regiones que experimentan eventos de remoción de árboles",
        "leg_loss_1": "🟢 Intacto", "leg_loss_2": "🟡 Menor", "leg_loss_3": "🟠 Moderado", "leg_loss_4": "🔴 Alto", "leg_loss_5": "🟣 Crítico",
        "leg_health_title": "🌿 Leyenda del Rastreador de Dosel Saludable",
        "leg_health_sub": "Mapeo de la biomasa de la vegetación y la densidad estructural",
        "leg_health_1": "🟢 Disperso", "leg_health_2": "🟡 Suelo", "leg_health_3": "🟠 Medio", "leg_health_4": "🔴 Denso", "leg_health_5": "🟣 Máximo",
        "leg_sprawl_title": "🏗️ Leyenda de Tendencias de Expansión Urbana",
        "leg_sprawl_sub": "Identificación de crecimiento estructural y pavimentación artificial",
        "leg_sprawl_1": "🟢 Rural", "leg_sprawl_2": "🟡 Bajo Construido", "leg_sprawl_3": "🟠 En Desarrollo", "leg_sprawl_4": "🔴 Alto Construido", "leg_sprawl_5": "🟣 Pavimentado",
        "leg_gen_title": "🔥 Leyenda de Intensidad del Mapa de Calor",
        "leg_gen_sub": "Diseño de mapeo del espectro meteorológico",
        "leg_gen_1": "🟢 Bajo", "leg_gen_2": "🟡 Moderado", "leg_gen_3": "🟠 Alto", "leg_gen_4": "🔴 Severe", "leg_gen_5": "🟣 Máximo"
    },
    "fr": {
        "app_title": "Analyse d'Imagerie Aérienne",
        "app_description": "Téléchargez des images satellites ou de drones pour analyser la végétation, l'eau, le sol, le sable et d'autres surfaces.",
        "upload_imagery": "Télécharger de l'imagerie aérienne",
        "supported_formats": "Formats pris en charge : JPEG, PNG, GeoTIFF",
        "imagery_view": "Vue de l'imagerie",
        "select_view_mode": "Sélectionnez le mode de vue :",
        "original_image": "Image originale",
        "segmented_view": "Vue segmentée",
        "segmented_view_caption": "Vue segmentée",
        "resolution": "Résolution",
        "segmented_view_not_available": "Vue segmentée non disponible ou image non encore traitée.",
        "uploaded_imagery_title": "Imagerie téléchargée",
        "upload_to_preview": "Téléchargez une image pour en voir un aperçu ici.",
        "area_breakdown": "Matrice de répartition des zones",
        "analysis_results": "Résultats de l'analyse",
        "vegetation": "Couverture de la canopée végétale",
        "water": "Corps de surface hydrologiques",
        "other": "Étalement urbain & infrastructures bâties",
        "soil": "Matrice de sol exposé",
        "sand": "Désertification & terres arides",
        "upload_to_run_analysis": "Téléchargez une image pour lancer l'analyse.",
        "images_analyzed": "Images distinctes traitées",
        "last_processed": "Dernièrement traitée",
        "terra_drone_title": "Terra Drone",
        "dashboard_caption": "Tableau de bord d'analyse d'imagerie aérienne",
        "processing_spinner": "Analyse des couches d'images...",
        "analysis_complete": "Analyse de l'image terminée.",
        "density_grids": "Grilles de densité (16x16)",
        "vegetation_density": "Densité de végétation (%)",
        "water_density": "Densité de l'eau (%)",
        "other_density": "Densité de l'étalement urbain (%)",
        "soil_density": "Densité du sol (%)",
        "sand_density": "Densité des terres arides (%)",
        "select_language": "Sélectionnez votre langue :",
        "welcome_message": "Bienvenue sur Terra Drone ! Veuillez sélectionner votre langue préférée pour continuer.",
        "session": "Stats de la session",
        "select_mode_title": "Sélectionner le mode de fonctionnement",
        "select_mode_desc": "Choisissez comment vous souhaitez traiter vos actifs aériens aujourd'hui :",
        "mode_single": "Analyse d'image unique",
        "mode_compare": "Mode de comparaison d'images",
        "upload_img_1": "Télécharger l'image de référence (Image 1)",
        "upload_img_2": "Télécharger l'image de comparaison (Image 2)",
        "comparison_dashboard": "Matrice de détection de changement & comparaison",
        "metric_label": "Classe d'entité",
        "img1_label": "Image 1 (%)",
        "img2_label": "Image 2 (%)",
        "diff_label": "Différence (Δ %)",
        # Newly Localized UI Elements
        "change_mode": "↩️ Changer de mode de fonctionnement",
        "full_app_reset": "🔄 Réinitialisation complète",
        "start_new_comparison": "🔄 Démarrer une nouvelle comparaison",
        "hotspot_settings": "🔥 Paramètres de la carte de chaleur",
        "select_hotspot_layer": "Sélectionner la couche de points chauds :",
        "overlay_intensity": "Intensité de la superposition",
        "image_1_baseline": "📷 Image 1 (Référence)",
        "image_2_comparison": "📷 Image 2 (Comparaison)",
        "view_raw_matrix": "📊 Afficher les grilles de densité matricielle",
        "show_mitigation": "📋 Afficher les suggestions d'atténuation",
        "mitigation_planner_title": "📋 Planificateur d'Intervention et d'Atténuation Environnementale",
        "biomass_loss": "Perte de biomasse détectée",
        "biomass_growth": "Croissance de la biomasse",
        "hydrological_loss": "Perte hydrologique",
        "stable_volume": "Volume stable",
        "critical_dryout": "🚨 ÉVÉNEMENT DE SÉCHERESSE CRITIQUE",
        "controlled_dev": "■ Seuil de développement contrôlé",
        "expansion_spike": "Pic d'étalement détecté",
        "topsoil_exposure": "▲ Augmentation de l'exposition du sol arable",
        "stabilized_surface": "Couverture de surface stabilisée",
        "card_veg": "Quantité de Végétation",
        "card_water": "Indicateur de Pénurie d'Eau",
        "card_sprawl": "Expansion des Infrastructures",
        "card_soil": "Matrice de Sol Exposé",
        "hotspot_none": "Aucun",
        "hotspot_urban": "Tendances de l'Étalement Urbain",
        "hotspot_loss": "Matrice de Perte de Canopée",
        "hotspot_healthy": "Suivi de la Canopée Saine",
        # SDG Localization
        "sdg_intro": "Bienvenue dans la <strong>Terra Drone Imagery Suite</strong>, une application d'intelligence avancée conçue pour transformer l'imagerie satellite, aérienne et de drone haute résolution en couches de données spatiales exploitables. En combinant des masques de classification automatiques par vision par ordinateur avec des structures de grille thermique haute fidélité, cette suite fournit des mesures de surveillance directes pour l'évaluation de l'impact environnemental, le suivi des changements écologiques et la planification des infrastructures.",
        "sdg_title": "🌿 Alignement avec l'Objectif de Développement Durable 11 de l'ONU",
        "sdg_desc": "L'<strong>ODD 11 : Villes et Communautés Durables</strong> vise à rendre les établissements humains inclusifs, sûrs, résilients et durables. Cette application soutient activement cette mission grâce à des fonctionnalités environnementales et géospatiales ciblées :",
        "sdg_target_1": "<strong>Cible 11.3 (Urbanisation inclusive & durable) :</strong> En surveillant les indices d'<strong>Étalement Urbain & Infrastructures Bâties</strong>, les urbanistes peuvent suivre les limites structurelles au fil du temps pour prévenir l'étalement horizontal incontrôlé et protéger les zones naturelles.",
        "sdg_target_2": "<strong>Cible 11.7 (Assurer l'accès à des espaces verts et publics) :</strong> La <strong>Matrice de Couverture de la Canopée Végétale</strong> haute performance fournit aux développeurs des mesures de couverture concrètes pour préserver, cartographier et étendre les espaces verts, atténuant ainsi l'effet d'îlot de chaleur urbain.",
        "sdg_target_3": "<strong>Cible 11.B (Réduction des risques de catastrophe & résilience environnementale) :</strong> En utilisant le <strong>Moteur de Détection de Changement par Comparaison d'Images</strong>, les régions peuvent évaluer activement les transformations environnementales, la dynamique de désertification et les vulnérabilités des eaux de surface afin d'ajuster les stratégies d'atténuation écologique.",
        "btn_select_single": "Sélectionner le mode unique",
        "btn_select_compare": "Sélectionner le mode de comparaison",
        # Mitigation Text
        "mit_veg_low_title": "**🚨 Profil de Faible Biomasse Déclenché**",
        "mit_veg_low_desc": "* Lancer des opérations de réensemencement ciblées ou des barrières agroforestières.\n* Restreindre le défrichement structurel dans les quadrants vulnérables.",
        "mit_veg_mod_title": "**⚠️ Distribution Modérée de la Canopée**",
        "mit_veg_mod_desc": "* Maintenance sur le terrain localisée requise.\n* Limiter les permis d'exploitation forestière dans les territoires tampons.",
        "mit_veg_opt_title": "**✅ Noyau de Densité de Biomasse Optimal Vérifié**",
        "mit_veg_opt_desc": "* Indices de stabilité naturelle normaux confirmés.",
        "mit_wat_low_title": "**⚠️ Alerte de Profil de Pénurie Hydrologique**",
        "mit_wat_low_desc": "* Déployer des barrières d'évaporation ou des restrictions de ruissellement sur les plans d'eau de surface.",
        "mit_wat_opt_title": "**💧 Tampon d'Eau Dans les Normes**",
        "mit_wat_opt_desc": "* La surveillance standard reste suffisante.",
        "mit_urb_high_title": "**🏗️ Forte Expansion des Infrastructures Détectée**",
        "mit_urb_high_desc": "* Imposer des configurations de matériaux perméables pour les nouvelles constructions.\n* Intégrer des micro-corridors forestiers urbains pour atténuer les effets thermiques.",
        "mit_urb_opt_title": "**🌲 Matrice d'Actifs Équilibrée entre Ville et Nature**",
        "mit_urb_opt_desc": "* Les limites de zonage sont saines.",
        "mit_arid_high_title": "**🏜️ Signal Actif de Désertification / Exposition du Sol**",
        "mit_arid_high_desc": "* Établir des structures de broussailles à racines profondes sur les limites sablonneuses.\n* Installer des barrières brise-vent perpendiculaires.",
        # Legends
        "leg_loss_title": "📉 Légende du Suivi de Perte de Canopée",
        "leg_loss_sub": "Ciblage des régions subissant des événements d'abattage d'arbres",
        "leg_loss_1": "🟢 Intacte", "leg_loss_2": "Mineure", "leg_loss_3": "Modérée", "leg_loss_4": "Élevée", "leg_loss_5": "Critique",
        "leg_health_title": "🌿 Légende du Suivi de Canopée Saine",
        "leg_health_sub": "Cartographie de la biomasse végétale et de la densité structurelle",
        "leg_health_1": "🟢 Éparse", "leg_health_2": "Sol", "leg_health_3": "Moyenne", "leg_health_4": "Dense", "leg_health_5": "Maximale",
        "leg_sprawl_title": "🏗️ Légende des Tendances de l'Étalement Urbain",
        "leg_sprawl_sub": "Identification de la croissance structurelle et du pavage artificiel",
        "leg_sprawl_1": "🟢 Rural", "leg_sprawl_2": "Faiblement Bâti", "leg_sprawl_3": "En Développement", "leg_sprawl_4": "Fortement Bâti", "leg_sprawl_5": "Pavé",
        "leg_gen_title": "🔥 Légende d'Intensité de la Carte de Chaleur",
        "leg_gen_sub": "Configuration de cartographie du spectre météorologique",
        "leg_gen_1": "🟢 Faible", "leg_gen_2": "Modérée", "leg_gen_3": "Élevée", "leg_gen_4": "Sévère", "leg_gen_5": "Maximale"
    },
    "de": {
        "app_title": "Luftbildanalyse",
        "app_description": "Laden Sie Drohnen- oder Satellitenbilder hoch, um Vegetation, Wasser, Boden, Sand und andere Oberflächen zu analysieren.",
        "upload_imagery": "Luftbilder hochladen",
        "supported_formats": "Unterstützte Formate: JPEG, PNG, GeoTIFF",
        "imagery_view": "Bildansicht",
        "select_view_mode": "Ansichtsmodus wählen:",
        "original_image": "Originalbild",
        "segmented_view": "Segmentierte Ansicht",
        "segmented_view_caption": "Segmentierte Ansicht",
        "resolution": "Auflösung",
        "segmented_view_not_available": "Segmentierte Ansicht nicht verfügbar oder Bild noch nicht verarbeitet.",
        "uploaded_imagery_title": "Hochgeladene Bilder",
        "upload_to_preview": "Laden Sie ein Bild hoch, um hier eine Vorschau anzuzeigen.",
        "area_breakdown": "Flächenverteilungsmatrix",
        "analysis_results": "Analyseergebnisse",
        "vegetation": "Vegetationsüberschneidung / Kronendach",
        "water": "Hydrologische Oberflächengewässer",
        "other": "Zersiedelung & bebaute Infrastruktur",
        "soil": "Freiliegende Bodenmatrix",
        "sand": "Desertifikation & trockenes Land",
        "upload_to_run_analysis": "Laden Sie ein Bild hoch, um die Analyse zu starten.",
        "images_analyzed": "Verarbeitete unterschiedliche Bilder",
        "last_processed": "Zuletzt verarbeitet",
        "terra_drone_title": "Terra Drone",
        "dashboard_caption": "Dashboard für Luftbildanalyse",
        "processing_spinner": "Bildschichten werden analysiert...",
        "analysis_complete": "Bildanalyse abgeschlossen.",
        "density_grids": "Dichtegitter (16x16)",
        "vegetation_density": "Vegetationsdichte (%)",
        "water_density": "Wasserdichte (%)",
        "other_density": "Zersiedelungsdichte (%)",
        "soil_density": "Bodendichte (%)",
        "sand_density": "Trockenlanddichte (%)",
        "select_language": "Wählen Sie Ihre Sprache:",
        "welcome_message": "Willkommen bei Terra Drone! Bitte wählen Sie Ihre bevorzugte Sprache aus, um fortzufahren.",
        "session": "Sitzungsstatistiken",
        "select_mode_title": "Betriebsmodus auswählen",
        "select_mode_desc": "Wählen Sie aus, wie Sie Ihre Luftbilddaten heute verarbeiten möchten:",
        "mode_single": "Einzelbildanalyse",
        "mode_compare": "Bildvergleichsmodus",
        "upload_img_1": "Basisbild hochladen (Bild 1)",
        "upload_img_2": "Vergleichsbild hochladen (Bild 2)",
        "comparison_dashboard": "Veränderungserkennung & Vergleichsmatrix",
        "metric_label": "Merkmalsklasse",
        "img1_label": "Bild 1 (%)",
        "img2_label": "Bild 2 (%)",
        "diff_label": "Differenz (Δ %)",
        # Newly Localized UI Elements
        "change_mode": "↩️ Betriebsmodus wechseln",
        "full_app_reset": "🔄 Vollständiger App-Reset",
        "start_new_comparison": "🔄 Neuen Vergleich starten",
        "hotspot_settings": "🔥 Hotspot-Heatmap-Einstellungen",
        "select_hotspot_layer": "Hotspot-Ebene auswählen:",
        "overlay_intensity": "Überlagerungsintensität",
        "image_1_baseline": "📷 Bild 1 (Basis)",
        "image_2_comparison": "📷 Bild 2 (Vergleich)",
        "view_raw_matrix": "📊 Rohdaten-Dichtegitter anzeigen",
        "show_mitigation": "📋 Maßnahmenvorschläge anzeigen",
        "mitigation_planner_title": "📋 Umweltinterventions- & Maßnahmenplaner",
        "biomass_loss": "Biomasseverlust erkannt",
        "biomass_growth": "Biomassewachstum",
        "hydrological_loss": "Hydrologischer Verlust",
        "stable_volume": "Stabiles Volumen",
        "critical_dryout": "🚨 KRITISCHES AUSTROCKNUNGSEREIGNIS",
        "controlled_dev": "■ Kontrollierte Entwicklungsgrenze",
        "expansion_spike": "Zersiedelungsspitze erkannt",
        "topsoil_exposure": "▲ Erhöhte Oberbodenfreilegung",
        "stabilized_surface": "Stabilisierte Oberflächenbedeckung",
        "card_veg": "Vegetationsmenge",
        "card_water": "Wasserknappheitsindikator",
        "card_sprawl": "Infrastruktur-Expansion",
        "card_soil": "Freiliegende Bodenmatrix",
        "hotspot_none": "Keine",
        "hotspot_urban": "Zersiedelungstrends",
        "hotspot_loss": "Kronendachverlust-Matrix",
        "hotspot_healthy": "Gesundes Kronendach-Tracking",
        # SDG Localization
        "sdg_intro": "Willkommen bei der <strong>Terra Drone Imagery Suite</strong>, einer hochentwickelten Intelligence-Anwendung zur Transformation hochauflösender Drohnen-, Luft- und Satellitenbilder in räumliche, handlungsrelevante Datenebenen. Durch die Kombination automatischer Computer-Vision-Klassifizierungsmasken mit hochpräzisen thermischen Gitterstrukturen liefert diese Suite direkte Überwachungsmetriken für die Umweltverträglichkeitsprüfung, ökologische Veränderungsverfolgung und übergeordnete Infrastrukturplanung.",
        "sdg_title": "🌿 Ausrichtung auf UN-Nachhaltigkeitsziel 11",
        "sdg_desc": "Das <strong>SDG 11: Nachhaltige Städte und Gemeinden</strong> zielt darauf ab, menschliche Siedlungen inklusiv, sicher, widerstandsfähig und nachhaltig zu machen. Diese Anwendung treibt diese Mission durch gezielte umweltbezogene und geo-spatiale Funktionen aktiv voran:",
        "sdg_target_1": "<strong>Ziel 11.3 (Inklusive & nachhaltige Urbanisierung):</strong> Durch die Überwachung der Indizes für <strong>Zersiedelung & bebaute Infrastruktur</strong> können Stadtplaner strukturelle Grenzen im Laufe der Zeit kontrollieren, um eine unkontrollierte horizontale Zersiedelung zu verhindern und Naturzonen zu schützen.",
        "sdg_target_2": "<strong>Ziel 11.7 (Zugang zu grünen & öffentlichen Räumen):</strong> Die hochleistungsfähige <strong>Vegetationsüberschneidungs-Matrix</strong> liefert Entwicklern konkrete Abdeckungsmetriken, um Grünflächen zu erhalten, zu kartieren und zu erweitern, wodurch der urbane Hitzeinseleffekt gemildert wird.",
        "sdg_target_3": "<strong>Ziel 11.B (Katastrophenrisikominderung & Umweltresilienz):</strong> Durch die Nutzung der <strong>Bildvergleichs-Veränderungserkennungs-Engine</strong> können Regionen Umweltveränderungen, Desertifikationsdynamiken und Anfälligkeiten des Oberflächenwassers aktiv bewerten, um ökologische Minderungsstrategien anzupassen.",
        "btn_select_single": "Einzelmodus auswählen",
        "btn_select_compare": "Vergleichsmodus auswählen",
        # Mitigation Text
        "mit_veg_low_title": "**🚨 Profil für niedrige Biomasse ausgelöst**",
        "mit_veg_low_desc": "* Starten Sie gezielte Wiederaufforstungsmaßnahmen oder agroforstwirtschaftliche Barrieren.\n* Beschränken Sie bauliche Rodungen in gefährdeten Quadranten.",
        "mit_veg_mod_title": "**⚠️ Moderate Kronendach-Biomasseverteilung**",
        "mit_veg_mod_desc": "* Lokale Feldpflege angefordert.\n* Beschränken Sie Holzeinschlagserlaubnisse in Puffergebieten.",
        "mit_veg_opt_title": "**✅ Optimaler Biomassedichtekern verifiziert**",
        "mit_veg_opt_desc": "* Normale natürliche Stabilitätsindizes bestätigt.",
        "mit_wat_low_title": "**⚠️ Hydrologischer Knappheitsprofil-Alarm**",
        "mit_wat_low_desc": "* Verdunstungsbarrieren oder Abflussbeschränkungen über Oberflächengewässern errichten.",
        "mit_wat_opt_title": "**💧 Wasserpuffer innerhalb der Standardbereiche**",
        "mit_wat_opt_desc": "* Standardmäßige Laufzeitüberwachung bleibt ausreichend.",
        "mit_urb_high_title": "**🏗️ Hohe Infrastruktur-Expansion festgestellt**",
        "mit_urb_high_desc": "* Durchsetzung durchlässiger Materialkonfigurationen für Neubauten.\n* Integration urbaner Mikro-Waldkorridore zur Abmilderung thermischer Effekte.",
        "mit_urb_opt_title": "**🌲 Ausgewogene Urban-zu-Natur-Asset-Matrix**",
        "mit_urb_opt_desc": "* Flächennutzungsgrenzen sind stabil.",
        "mit_arid_high_title": "**🏜️ Aktives Desertifikations- / Bodenfreilegungssignal**",
        "mit_arid_high_desc": "* Tiefwurzelnde Buschstrukturen an sandigen Grenzen etablieren.\n* Rechtwinklige Windschutzstreifen installieren.",
        # Legends
        "leg_loss_title": "📉 Legende für Kronendachverlust-Tracker",
        "leg_loss_sub": "Zielgebiete mit Baumfällungsereignissen",
        "leg_loss_1": "🟢 Intakt", "leg_loss_2": "Gering", "leg_loss_3": "Moderat", "leg_loss_4": "Hoch", "leg_loss_5": "Kritisch",
        "leg_health_title": "🌿 Legende für gesundes Kronendach-Tracking",
        "leg_health_sub": "Kartierung von Vegetationsbiomasse und Strukturdichte",
        "leg_health_1": "🟢 Spärlich", "leg_health_2": "Boden", "leg_health_3": "Mittel", "leg_health_4": "Dicht", "leg_health_5": "Maximum",
        "leg_sprawl_title": "🏗️ Legende für Zersiedelungstrends",
        "leg_sprawl_sub": "Identifizierung von Strukturwachstum und künstlicher Pflasterung",
        "leg_sprawl_1": "🟢 Ländlich", "leg_sprawl_2": "Gering bebaut", "leg_sprawl_3": "In Entwicklung", "leg_sprawl_4": "Stark bebaut", "leg_sprawl_5": "Gepflastert",
        "leg_gen_title": "🔥 Legende für Heatmap-Intensität",
        "leg_gen_sub": "Wetterspektrum-Mapping-Layout",
        "leg_gen_1": "🟢 Niedrig", "leg_gen_2": "Moderat", "leg_gen_3": "Hoch", "leg_gen_4": "Schwer", "leg_gen_5": "Maximum"
    },
    "zh": {
        "app_title": "航拍图像分析",
        "app_description": "上传无人机或卫星图像以分析植被、水体、土壤、沙地和其他地表。",
        "upload_imagery": "上传航拍图像",
        "supported_formats": "支持的格式：JPEG, PNG, GeoTIFF",
        "imagery_view": "图像视图",
        "select_view_mode": "选择视图模式：",
        "original_image": "原始图像",
        "segmented_view": "分割视图",
        "segmented_view_caption": "分割视图",
        "resolution": "分辨率",
        "segmented_view_not_available": "分割视图不可用或图像尚未处理。",
        "uploaded_imagery_title": "已上传图像",
        "upload_to_preview": "上传图像以在此处预览。",
        "area_breakdown": "面积细分矩阵",
        "analysis_results": "分析结果",
        "vegetation": "植被冠层覆盖",
        "water": "水文地表水体",
        "other": "城市扩张与建筑基础设施",
        "soil": "裸露土壤矩阵",
        "sand": "土地荒漠化与干旱地",
        "upload_to_run_analysis": "上传图像以运行分析。",
        "images_analyzed": "已处理的独立图像数",
        "last_processed": "最后处理时间",
        "terra_drone_title": "Terra Drone",
        "dashboard_caption": "航拍图像分析仪表板",
        "processing_spinner": "正在分析图像图层...",
        "analysis_complete": "图像 analysis 完成。",
        "density_grids": "密度网格 (16x16)",
        "vegetation_density": "植被密度 (%)",
        "water_density": "水体密度 (%)",
        "other_density": "城市扩张密度 (%)",
        "soil_density": "土壤密度 (%)",
        "sand_density": "荒漠化土地密度 (%)",
        "select_language": "选择您的语言：",
        "welcome_message": "欢迎使用 Terra Drone！请选择您的首选语言以继续。",
        "session": "会话统计",
        "select_mode_title": "选择运行模式",
        "select_mode_desc": "选择您今天希望如何处理您的航拍资产：",
        "mode_single": "单图分析",
        "mode_compare": "图像对比模式",
        "upload_img_1": "上传基准图像 (图像 1)",
        "upload_img_2": "上传对比图像 (图像 2)",
        "comparison_dashboard": "变化检测与对比矩阵",
        "metric_label": "特征类别",
        "img1_label": "图像 1 (%)",
        "img2_label": "图像 2 (%)",
        "diff_label": "差异 (Δ %)",
        # Newly Localized UI Elements
        "change_mode": "↩️ 更改运行模式",
        "full_app_reset": "🔄 重置整个应用",
        "start_new_comparison": "🔄 开启全新对比",
        "hotspot_settings": "🔥 热力图参数设置",
        "select_hotspot_layer": "选择热力图分析图层：",
        "overlay_intensity": "热力图覆盖透明度",
        "image_1_baseline": "📷 图像 1 (基准线)",
        "image_2_comparison": "📷 图像 2 (对比图)",
        "view_raw_matrix": "📊 查看原始网格密度矩阵数据",
        "show_mitigation": "📋 显示生态干预与改善建议",
        "mitigation_planner_title": "📋 环境干预与生态改善计划器",
        "biomass_loss": "检测到植被生物量减少",
        "biomass_growth": "植被生物量正向增长",
        "hydrological_loss": "地表水体流失缩减",
        "stable_volume": "水体蓄水量平稳",
        "critical_dryout": "🚨 严重干涸干旱事件预警",
        "controlled_dev": "■ 建筑边界受控发展状态",
        "expansion_spike": "检测到城市化急剧扩张",
        "topsoil_exposure": "▲ 表层土壤裸露率增加",
        "stabilized_surface": "地表覆盖维持平稳",
        "card_veg": "植被覆被总量",
        "card_water": "水资源稀缺指数",
        "card_sprawl": "基础设施化扩张",
        "card_soil": "裸露土壤矩阵",
        "hotspot_none": "无",
        "hotspot_urban": "城市化扩张趋势",
        "hotspot_loss": "森林冠层流失矩阵",
        "hotspot_healthy": "健康植被冠层追踪",
        # SDG Localization
        "sdg_intro": "欢迎使用 <strong>Terra Drone 遥感图像智能套件</strong>。这是一款专为将高分辨率无人机航拍、航空和卫星图像转化为空间结构化、可付诸行动的数据图层而设计的尖端智能应用。系统结合了计算机视觉全自动分类掩膜与高保真热力网格，为环境影响评估、生态环境变化追踪及城市综合基础设施规划提供直观的数据监测指标。",
        "sdg_title": "🌿 深度契合联合国可持续发展目标 11 (SDG 11)",
        "sdg_desc": "<strong>SDG 11：可持续城市和社区</strong> 旨在使人类聚居地具备包容性、安全性、韧性和可持续性。本应用通过精准的地理空间与环境监控功能，深度助力该目标的实现：",
        "sdg_target_1": "<strong>细分目标 11.3（包容性及可持续的城市化）：</strong> 通过动态监测 <strong>城市扩张与建筑基础设施</strong> 指数，城市规划者可跨时间跨度严格审计建筑边界，从而遏制失控的无序横向扩张并保护原生自然生态区。",
        "sdg_target_2": "<strong>细分目标 11.7（提供安全的绿色公共空间）：</strong> 高性能的 <strong>植被冠层覆盖矩阵</strong> 为开发商与环保机构提供具体的绿化覆盖率基准，科学指导保留、规划与扩大城市绿地，以此减缓城市热岛效应。",
        "sdg_target_3": "<strong>细分目标 11.B（提高防灾减灾与环境韧性）：</strong> 借助 <strong>图像对比与变化检测引擎</strong>，各地区能实时评估地表形态异动、土地荒漠化演变及地表水源脆弱性，从而动态调整个态治理与灾害缓解策略。",
        "btn_select_single": "选择单图分析模式",
        "btn_select_compare": "选择对比模式",
        # Mitigation Text
        "mit_veg_low_title": "**🚨 触发低生物量植被警报**",
        "mit_veg_low_desc": "* 立即启动靶向飞播结种或建设复合型农林复合屏障。\n* 严格限制生态脆弱象限内的结构性砍伐。",
        "mit_veg_mod_title": "**⚠️ 植被冠层生物量分布中等**",
        "mit_veg_mod_desc": "* 建议对受影响区域进行局部实地养护。\n* 限制生态缓冲区内的商业采伐许可。",
        "mit_veg_opt_title": "**✅ 通过最佳生物量密度核心验证**",
        "mit_veg_opt_desc": "* 自然生态稳定性指标处于健康区间。",
        "mit_wat_low_title": "**⚠️ 触发地表水资源匮乏警报**",
        "mit_wat_low_desc": "* 针对现有露天蓄水体布设蒸发阻隔层，或实施径流限制性保护。",
        "mit_wat_opt_title": "**💧 水资源蓄水能力处于标准区间**",
        "mit_wat_opt_desc": "* 维持标准周期的常规环境监测即可。",
        "mit_urb_high_title": "**🏗️ 监测到高强度基础设施扩张**",
        "mit_urb_high_desc": "* 对新规划区域强制实施透水性铺装建材规范。\n* 积极植入城市微型森林生态廊道以削减热岛效应。",
        "mit_urb_opt_title": "**🌲 人类建筑与自然资源比例结构平衡**",
        "mit_urb_opt_desc": "* 土地分区红线执行状态良好。",
        "mit_arid_high_title": "**🏜️ 捕获到持续荒漠化/土壤大量裸露信号**",
        "mit_arid_high_desc": "* 在沙化边缘带科学引种深根系固沙灌木丛。\n* 配置垂直于盛行风向的生态防风林带。",
        # Legends
        "leg_loss_title": "📉 森林冠层流失追踪图例",
        "leg_loss_sub": "精确定位发生树木移除与砍伐的重点林区",
        "leg_loss_1": "🟢 完好无损", "leg_loss_2": "轻微变化", "leg_loss_3": "中度流失", "leg_loss_4": "高度破坏", "leg_loss_5": "关键极值",
        "leg_health_title": "🌿 健康植被冠层分布图例",
        "leg_health_sub": "映射地表植被生物量总量与植物空间结构密度",
        "leg_health_1": "🟢 稀疏灌丛", "leg_health_2": "低矮地被", "leg_health_3": "中等郁闭", "leg_health_4": "茂密森林", "leg_health_5": "峰值茂盛",
        "leg_sprawl_title": "🏗️ 城市无序扩张趋势图例",
        "leg_sprawl_sub": "识别硬化地表增长、建筑覆盖及人工铺装演变",
        "leg_sprawl_1": "🟢 乡村风貌", "leg_sprawl_2": "低密建筑", "leg_sprawl_3": "发展建设中", "leg_sprawl_4": "高密硬化", "leg_sprawl_5": "全面硬化",
        "leg_gen_title": "🔥 热力图频谱辐射图例",
        "leg_gen_sub": "基于气象雷达谱系高对比度颜色映射",
        "leg_gen_1": "🟢 低阶辐射", "leg_gen_2": "中等辐射", "leg_gen_3": "高阶辐射", "leg_gen_4": "严重辐射", "leg_gen_5": "核心峰值"
    }
}

def reset_entire_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def clear_comparison_cache() -> None:
    """Clears all image assets and increments keys to force drop uploaded files from UI."""
    st.session_state.compare_img1 = None
    st.session_state.compare_img2 = None
    st.session_state.compare_res1 = None
    st.session_state.compare_res2 = None
    st.session_state.compare_seg1 = None
    st.session_state.compare_seg2 = None
    st.session_state.compare_grids1 = None
    st.session_state.compare_grids2 = None
    st.session_state.processed_f1 = None
    st.session_state.processed_f2 = None
    st.session_state.comparison_run_id += 1
    st.rerun()

def init_session_state() -> None:
    defaults = {
        "uploaded_image": None,
        "image_name": None,
        "ai_processed": False,
        "processing_done": False,
        "analysis_results": None,
        "segmented_image": None,
        "density_grids": None,
        "last_processed_at": None,
        "images_analyzed": 0,
        "selected_language": None,
        "selected_mode": None,  
        "compare_img1": None,
        "compare_img2": None,
        "compare_res1": None,
        "compare_res2": None,
        "compare_seg1": None,
        "compare_seg2": None,
        "compare_grids1": None,
        "compare_grids2": None,
        "processed_single_name": None,
        "processed_f1": None,
        "processed_f2": None,
        "comparison_run_id": 0,
        "show_planner_single": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def analyze_image_pixels(image: Image.Image) -> tuple[dict, np.ndarray, dict]:
    img_array = np.array(image)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    height, width, _ = img_array.shape
    total_pixels = height * width

    veg_p = 0
    wat_p = 0
    oth_p = 0
    soil_p = 0
    sand_p = 0

    classification_map = np.zeros((height, width), dtype=np.uint8)

    VEGETATION = 0
    WATER = 1
    OTHER = 2
    SOIL = 3
    SAND = 4

    grid_size = 16
    cell_height = height // grid_size
    cell_width = width // grid_size

    temp_veg_counts = np.zeros((grid_size, grid_size), dtype=int)
    temp_water_counts = np.zeros((grid_size, grid_size), dtype=int)
    temp_other_counts = np.zeros((grid_size, grid_size), dtype=int)
    temp_soil_counts = np.zeros((grid_size, grid_size), dtype=int)
    temp_sand_counts = np.zeros((grid_size, grid_size), dtype=int)

    for y in range(height):
        for x in range(width):
            r, g, b = map(int, img_array[y, x])

            grid_y = min(y // cell_height, grid_size - 1)
            grid_x = min(x // cell_width, grid_size - 1)

            if (g > r + 2 and g > b + 2 and g < 220) or \
               (g > r and g >= b - 2 and 40 < g < 95) or \
               (45 < r < 110 and 45 < g < 105 and 50 < b < 120 and abs(r - g) <= 12 and g > b - 15 and r > b - 15):
                veg_p += 1
                classification_map[y, x] = VEGETATION
                temp_veg_counts[grid_y, grid_x] += 1
                
            elif b > r + 22 and b > g + 16 and b > 45:
                if abs(r - g) <= 10 and (b - g) < 45:
                    oth_p += 1
                    classification_map[y, x] = OTHER
                    temp_other_counts[grid_y, grid_x] += 1
                else:
                    wat_p += 1
                    classification_map[y, x] = WATER
                    temp_water_counts[grid_y, grid_x] += 1

            elif (r > g + 10 and g >= b and 40 < r <= 130) or \
                 (115 < r < 185 and 90 < g < 155 and 70 < b < 130 and r > g + 16 and g > b + 5):
                soil_p += 1
                classification_map[y, x] = SOIL
                temp_soil_counts[grid_y, grid_x] += 1
            
            elif (130 < r < 210) and (115 < g < 190) and (90 < b < 165) and (r > g + 8) and (g > b + 8):
                sand_p += 1
                classification_map[y, x] = SAND
                temp_sand_counts[grid_y, grid_x] += 1
                
            else:
                oth_p += 1
                classification_map[y, x] = OTHER
                temp_other_counts[grid_y, grid_x] += 1

    vegetation_grid = np.zeros((grid_size, grid_size), dtype=float)
    water_grid = np.zeros((grid_size, grid_size), dtype=float)
    other_grid = np.zeros((grid_size, grid_size), dtype=float)
    soil_grid = np.zeros((grid_size, grid_size), dtype=float)
    sand_grid = np.zeros((grid_size, grid_size), dtype=float)

    for i in range(grid_size):
        for j in range(grid_size):
            actual_cell_pixels = 0
            for cy in range(i * cell_height, min((i + 1) * cell_height, height)):
                for cx in range(j * cell_width, min((j + 1) * cell_width, width)):
                    actual_cell_pixels += 1
            
            if actual_cell_pixels > 0:
                vegetation_grid[i, j] = (temp_veg_counts[i, j] / actual_cell_pixels) * 100
                water_grid[i, j] = (temp_water_counts[i, j] / actual_cell_pixels) * 100
                other_grid[i, j] = (temp_other_counts[i, j] / actual_cell_pixels) * 100
                soil_grid[i, j] = (temp_soil_counts[i, j] / actual_cell_pixels) * 100
                sand_grid[i, j] = (temp_sand_counts[i, j] / actual_cell_pixels) * 100

    analysis_results = {
        "vegetation_pct": (veg_p / total_pixels) * 100,
        "water_pct": (wat_p / total_pixels) * 100,
        "other_pct": (oth_p / total_pixels) * 100,
        "soil_pct": (soil_p / total_pixels) * 100,
        "sand_pct": (sand_p / total_pixels) * 100,
    }

    density_grids = {
        "vegetation_grid": vegetation_grid,
        "water_grid": water_grid,
        "other_grid": other_grid,
        "soil_grid": soil_grid,
        "sand_grid": sand_grid,
    }

    return analysis_results, classification_map, density_grids

def generate_segmented_image(classification_map: np.ndarray, original_image: Image.Image) -> Image.Image:
    height, width = classification_map.shape

    if original_image.mode != 'RGBA':
        original_image = original_image.convert('RGBA')
    original_array = np.array(original_image)

    overlay_array = np.zeros((height, width, 4), dtype=np.uint8)
    alpha = 140 

    COLOR_VEGETATION = [46, 204, 113, alpha]      
    COLOR_WATER = [41, 128, 185, alpha]           
    COLOR_OTHER = [149, 165, 166, alpha]  
    COLOR_SOIL = [139, 69, 19, alpha]             
    COLOR_SAND = [215, 185, 150, alpha] 

    for y in range(height):
        for x in range(width):
            val = classification_map[y, x]
            if val == 0: 
                overlay_array[y, x] = COLOR_VEGETATION
            elif val == 1: 
                overlay_array[y, x] = COLOR_WATER
            elif val == 2: 
                overlay_array[y, x] = COLOR_OTHER
            elif val == 3:
                overlay_array[y, x] = COLOR_SOIL
            else:
                overlay_array[y, x] = COLOR_SAND

    base_image = Image.fromarray(original_array, 'RGBA')
    overlay_image = Image.fromarray(overlay_array, 'RGBA')
    blended_image = Image.alpha_composite(base_image, overlay_image)

    return blended_image

def get_weather_colors_vectorized(intensities: np.ndarray) -> np.ndarray:
    stops = [
        (0.0,  [16, 185, 129]),   # Emerald Green
        (0.25, [234, 179, 8]),    # Pure Yellow
        (0.5,  [249, 115, 22]),   # Orange
        (0.75, [220, 38, 38]),    # Crimson Red
        (1.0,  [107, 33, 168])    # Dark Purple
    ]
    
    out = np.zeros(intensities.shape + (3,), dtype=np.uint8)
    intensities = np.clip(intensities, 0.0, 1.0)
    
    for i in range(len(stops) - 1):
        s1, c1 = stops[i]
        s2, c2 = stops[i+1]
        
        mask = (intensities >= s1) & (intensities <= s2)
        if not np.any(mask):
            continue
            
        ratio = (intensities[mask] - s1) / (s2 - s1)
        ratio = ratio[..., np.newaxis]
        
        c1_arr = np.array(c1)
        c2_arr = np.array(c2)
        
        color_vals = c1_arr + ratio * (c2_arr - c1_arr)
        out[mask] = color_vals.astype(np.uint8)
        
    return out

def generate_hotspot_overlay(original_image: Image.Image, density_grid: np.ndarray, target_layer: str, alpha: float) -> Image.Image:
    if target_layer == "None" or alpha == 0.0:
        return original_image

    if original_image.mode != "RGBA":
        original_image = original_image.convert("RGBA")
        
    width, height = original_image.size
    grid_size = 16
    cell_w = width / grid_size
    cell_h = height / grid_size
    
    overlay_mask = np.zeros((height, width, 4), dtype=np.uint8)
    max_radius = max(cell_w, cell_h) * 1.5
    
    r_int = int(np.ceil(max_radius))
    y_indices = np.arange(-r_int, r_int + 1)
    x_indices = np.arange(-r_int, r_int + 1)
    mesh_dx, mesh_dy = np.meshgrid(x_indices, y_indices)
    mesh_dist = np.sqrt(mesh_dx**2 + mesh_dy**2)
    
    valid_dist_mask = mesh_dist < max_radius
    fade_template = 0.5 * (1.0 + np.cos(np.pi * mesh_dist / max_radius))
    fade_template[~valid_dist_mask] = 0

    for i in range(grid_size):
        center_y = int((i + 0.5) * cell_h)
        y_start = max(0, center_y - r_int)
        y_end = min(height, center_y + r_int + 1)
        slice_y_start = r_int - (center_y - y_start)
        slice_y_end = r_int + (y_end - center_y)
        
        for j in range(grid_size):
            intensity = density_grid[i, j] / 100.0
            if intensity <= 0.02:
                continue
                
            center_x = int((j + 0.5) * cell_w)
            x_start = max(0, center_x - r_int)
            x_end = min(width, center_x + r_int + 1)
            slice_x_start = r_int - (center_x - x_start)
            slice_x_end = r_int + (x_end - center_x)
            
            fade_slice = fade_template[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
            combined_intensity = intensity * fade_slice
            
            active_pixels = combined_intensity > 0.02
            if not np.any(active_pixels):
                continue
                
            calculated_alphas = np.clip(combined_intensity * alpha * 255, 0, 255).astype(np.uint8)
            current_alphas = overlay_mask[y_start:y_end, x_start:x_end, 3]
            better_pixels = (calculated_alphas > current_alphas) & active_pixels
            
            if np.any(better_pixels):
                rgb_colors = get_weather_colors_vectorized(combined_intensity)
                
                target_rgb = overlay_mask[y_start:y_end, x_start:x_end, 0:3]
                target_alpha = overlay_mask[y_start:y_end, x_start:x_end, 3]
                
                target_rgb[better_pixels] = rgb_colors[better_pixels]
                target_alpha[better_pixels] = calculated_alphas[better_pixels]

    overlay_image = Image.fromarray(overlay_mask, "RGBA")
    return Image.alpha_composite(original_image, overlay_image)

def render_weather_legend_html(mode_index: int, t: dict) -> None:
    if mode_index == 2: # Loss
        title = t["leg_loss_title"]
        subtitle = t["leg_loss_sub"]
        l1, l2, l3, l4, l5 = t["leg_loss_1"], t["leg_loss_2"], t["leg_loss_3"], t["leg_loss_4"], t["leg_loss_5"]
    elif mode_index == 3: # Health
        title = t["leg_health_title"]
        subtitle = t["leg_health_sub"]
        l1, l2, l3, l4, l5 = t["leg_health_1"], t["leg_health_2"], t["leg_health_3"], t["leg_health_4"], t["leg_health_5"]
    elif mode_index == 1: # Sprawl
        title = t["leg_sprawl_title"]
        subtitle = t["leg_sprawl_sub"]
        l1, l2, l3, l4, l5 = t["leg_sprawl_1"], t["leg_sprawl_2"], t["leg_sprawl_3"], t["leg_sprawl_4"], t["leg_sprawl_5"]
    else:
        title = t["leg_gen_title"]
        subtitle = t["leg_gen_sub"]
        l1, l2, l3, l4, l5 = t["leg_gen_1"], t["leg_gen_2"], t["leg_gen_3"], t["leg_gen_4"], t["leg_gen_5"]

    legend_html = f"""
    <div style="font-family: 'Segoe UI', system-ui, sans-serif; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 20px; margin-top: 15px; margin-bottom: 5px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
            <span>{title}</span>
            <span style="font-size: 0.85rem; font-weight: 400; color: #64748b;">{subtitle}</span>
        </div>
        <div style="height: 14px; background: linear-gradient(to right, #10b981 0%, #eab308 25%, #f97316 50%, #dc2626 75%, #6b21a8 100%); border-radius: 4px; width: 100%;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 1.1rem; font-weight: 800; margin-top: 8px; color: #334155;">
            <span style="color: #10b981;">{l1}</span>
            <span style="color: #b45309;">{l2}</span>
            <span style="color: #c2410c;">{l3}</span>
            <span style="color: #b91c1c;">{l4}</span>
            <span style="color: #581c87;">{l5}</span>
        </div>
    </div>
    """
    components.html(legend_html, height=100, scrolling=False)

def render_dynamic_mitigation_suggestions(res: dict, t: dict) -> None:
    st.write("---")
    st.markdown(f"### {t['mitigation_planner_title']}")

    veg = res["vegetation_pct"]
    sprawl = res["other_pct"]
    arid = res["sand_pct"]
    water = res["water_pct"]

    col1, col2 = st.columns(2)

    with col1:
        if veg < 25.0:
            st.error(f"{t['mit_veg_low_title']}\n{t['mit_veg_low_desc']}")
        elif 25.0 <= veg < 50.0:
            st.warning(f"{t['mit_veg_mod_title']}\n{t['mit_veg_mod_desc']}")
        else:
            st.success(f"{t['mit_veg_opt_title']}\n{t['mit_veg_opt_desc']}")

        if water < 5.0:
            st.warning(f"{t['mit_wat_low_title']}\n{t['mit_wat_low_desc']}")
        else:
            st.info(f"{t['mit_wat_opt_title']}\n{t['mit_wat_opt_desc']}")

    with col2:
        if sprawl > 40.0:
            st.error(f"{t['mit_urb_high_title']}\n{t['mit_urb_high_desc']}")
        else:
            st.success(f"{t['mit_urb_opt_title']}\n{t['mit_urb_opt_desc']}")

        if arid > 20.0:
            st.warning(f"{t['mit_arid_high_title']}\n{t['mit_arid_high_desc']}")

def render_language_selection() -> None:
    st.title("Terra Drone Imagery Suite")
    st.write("Please select your preferred operational language / Por favor seleccione su idioma de operación:")

    lang_options = list(LANGUAGES.values())
    selected_lang_name = st.radio("Language / Idioma / Langue / Sprache", options=lang_options, index=0)
    selected_lang_key = next(key for key, name in LANGUAGES.items() if name == selected_lang_name)

    if st.button("Continue"):
        st.session_state.selected_language = selected_lang_key
        st.rerun()

def render_mode_selection(t: dict) -> None:
    st.title(f"🛰️ {t['terra_drone_title']}")
    
    st.markdown(f"""
    <div style="font-size: 1.25rem; line-height: 1.6; color: #1e293b; margin-bottom: 25px; font-family: system-ui, -apple-system, sans-serif;">
        {t['sdg_intro']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color: #eff6ff; border-left: 5px solid #3b82f6; padding: 20px; border-radius: 4px; margin-bottom: 25px; font-family: system-ui, -apple-system, sans-serif;">
        <h3 style="margin-top: 0; color: #1e3a8a; font-size: 1.45rem; font-weight: 700; margin-bottom: 12px;">{t['sdg_title']}</h3>
        <p style="font-size: 1.25rem; line-height: 1.6; color: #1e293b; margin-bottom: 14px;">
            {t['sdg_desc']}
        </p>
        <ul style="margin-bottom: 0; padding-left: 20px;">
            <li style="font-size: 1.25rem; line-height: 1.6; color: #1e293b; margin-bottom: 10px;">
                {t['sdg_target_1']}
            </li>
            <li style="font-size: 1.25rem; line-height: 1.6; color: #1e293b; margin-bottom: 10px;">
                {t['sdg_target_2']}
            </li>
            <li style="font-size: 1.25rem; line-height: 1.6; color: #1e293b;">
                {t['sdg_target_3']}
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")
    st.subheader(t["select_mode_title"])
    st.markdown(t["select_mode_desc"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 📊 {t['mode_single']}")
        st.caption("Extract land surface category distributions, grid matrices, and generate dynamic remediation tasks.")
        if st.button(t["btn_select_single"], use_container_width=True, type="primary"):
            st.session_state.selected_mode = "single"
            st.rerun()
    with col2:
        st.markdown(f"### 🔄 {t['mode_compare']}")
        st.caption("Upload distinct intervals to perform automatic subtraction loops and audit historic ground shifts.")
        if st.button(t["btn_select_compare"], use_container_width=True, type="primary"):
            st.session_state.selected_mode = "compare"
            st.rerun()

def render_sidebar(t: dict) -> None:
    st.sidebar.title(t["terra_drone_title"])
    st.sidebar.caption(t["dashboard_caption"])
    st.sidebar.divider()

    if st.sidebar.button(t["change_mode"], use_container_width=True):
        st.session_state.selected_mode = None
        st.rerun()
        
    if st.sidebar.button(t["full_app_reset"], use_container_width=True, type="secondary"):
        reset_entire_session()
        
    st.sidebar.divider()
    st.sidebar.subheader(t["session"])
    st.sidebar.metric(t["images_analyzed"], st.session_state.images_analyzed)

def render_main(t: dict) -> None:
    st.html("""
        <style>
            div[data-testid="stRadio"] label p {
                font-size: 1.05rem !important;
                font-weight: 600 !important;
                color: #0f172a !important;
            }
        </style>
    """)

    st.title(t["app_title"])
    st.markdown(t["app_description"])

    uploaded = st.file_uploader(t["upload_imagery"], type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded is not None:
        if st.session_state.processed_single_name != uploaded.name:
            st.session_state.images_analyzed += 1
            st.session_state.processed_single_name = uploaded.name
            st.session_state.show_planner_single = False

        st.session_state.uploaded_image = Image.open(io.BytesIO(uploaded.read()))
        st.session_state.image_name = uploaded.name
        with st.spinner(t["processing_spinner"]):
            analysis_results, classification_map, density_grids = analyze_image_pixels(st.session_state.uploaded_image)
            st.session_state.analysis_results = analysis_results
            st.session_state.segmented_image = generate_segmented_image(classification_map, st.session_state.uploaded_image)
            st.session_state.density_grids = density_grids
            st.session_state.processing_done = True
        st.success(t["analysis_complete"])

    st.divider()
    map_col, side_col = st.columns([1, 1])

    with map_col:
        if st.session_state.uploaded_image is not None:
            view_mode = st.radio(t["select_view_mode"], (t["original_image"], t["segmented_view"]), horizontal=True)
            if view_mode == t["original_image"]:
                st.image(st.session_state.uploaded_image, use_container_width=True)
            else:
                st.image(st.session_state.segmented_image, use_container_width=True)
                
            w, h = st.session_state.uploaded_image.size
            st.caption(f"{t['resolution']}: {w} × {h} px")

            if st.session_state.density_grids is not None:
                st.write("---")
                st.subheader(t["density_grids"])
                
                grids = st.session_state.density_grids
                veg_grid_df = pd.DataFrame(grids["vegetation_grid"])
                water_grid_df = pd.DataFrame(grids["water_grid"])
                other_grid_df = pd.DataFrame(grids["other_grid"])
                soil_grid_df = pd.DataFrame(grids["soil_grid"])
                sand_grid_df = pd.DataFrame(grids["sand_grid"])

                grid_style = [
                    {"selector": "td", "props": [("font-size", "13px"), ("font-weight", "bold"), ("text-align", "center")]},
                    {"selector": "th", "props": [("font-size", "11px")]}
                ]

                with st.expander(t["view_raw_matrix"], expanded=False):
                    st.write(f"**{t['vegetation_density']}**")
                    st.dataframe(veg_grid_df.style.set_table_styles(grid_style).background_gradient(cmap='Greens', vmin=0, vmax=100).format("{:.1f}"), use_container_width=True)

                    st.write(f"**{t['water_density']}**")
                    st.dataframe(water_grid_df.style.set_table_styles(grid_style).background_gradient(cmap='Blues', vmin=0, vmax=100).format("{:.1f}"), use_container_width=True)

                    st.write(f"**{t['other_density']}**")
                    st.dataframe(other_grid_df.style.set_table_styles(grid_style).background_gradient(cmap='Greys', vmin=0, vmax=100).format("{:.1f}"), use_container_width=True)

                    st.write(f"**{t['soil_density']}**")
                    st.dataframe(soil_grid_df.style.set_table_styles(grid_style).background_gradient(cmap='YlOrBr', vmin=0, vmax=100).format("{:.1f}"), use_container_width=True)

                    st.write(f"**{t['sand_density']}**")
                    st.dataframe(sand_grid_df.style.set_table_styles(grid_style).background_gradient(cmap='YlOrRd', vmin=0, vmax=100).format("{:.1f}"), use_container_width=True)
        else:
            st.info(t["upload_to_preview"])

    with side_col:
        if st.session_state.processing_done and st.session_state.analysis_results is not None:
            st.subheader(t["area_breakdown"])
            res = st.session_state.analysis_results
            
            labels_clean = [
                t["vegetation"],
                t["water"],
                t["other"],
                t["soil"],
                t["sand"]
            ]
            
            df_pie = pd.DataFrame({
                "Category": labels_clean,
                "Percentage": [res["vegetation_pct"], res["water_pct"], res["other_pct"], res["soil_pct"], res["sand_pct"]]
            })
            
            fig = px.pie(
                df_pie, 
                names="Category", 
                values="Percentage", 
                hole=0.45,
                color="Category", 
                color_discrete_map={
                    labels_clean[0]: "#2ecc71", 
                    labels_clean[1]: "#2980b9", 
                    labels_clean[2]: "#95a5a6",
                    labels_clean[3]: "#8b4513",
                    labels_clean[4]: "#d7b996"
                }
            )

            fig.update_layout(
                height=520, 
                margin=dict(l=20, r=20, t=20, b=100),
                font=dict(size=13),
                legend=dict(
                    font=dict(size=15),
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="center",
                    x=0.5
                )
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent',
                textfont=dict(size=16, color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("")
            if st.button(t["show_mitigation"], use_container_width=True, type="primary"):
                st.session_state.show_planner_single = not st.session_state.show_planner_single
            
            if st.session_state.show_planner_single:
                render_dynamic_mitigation_suggestions(res, t)

def render_comparison(t: dict) -> None:
    st.html("""
        <style>
            div[data-testid="stRadio"] label p {
                font-size: 1.05rem !important;
                font-weight: 600 !important;
                color: #0f172a !important;
            }
        </style>
    """)

    st.title(f"🔄 {t['mode_compare']}")
    st.markdown("Upload two captures to map changing ground classifications.")

    if st.button(t["start_new_comparison"], type="secondary"):
        clear_comparison_cache()

    st.sidebar.markdown(f"### {t['hotspot_settings']}")
    
    layer_map_labels = [t["hotspot_none"], t["hotspot_urban"], t["hotspot_loss"], t["hotspot_healthy"]]
    hotspot_label_selected = st.sidebar.selectbox(
        t["select_hotspot_layer"],
        options=layer_map_labels
    )
    hotspot_index = layer_map_labels.index(hotspot_label_selected)
    overlay_intensity = st.sidebar.slider(t["overlay_intensity"], min_value=0.0, max_value=1.0, value=0.60, step=0.05)

    run_id = st.session_state.comparison_run_id
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        f1 = st.file_uploader(t["upload_img_1"], type=["jpg", "jpeg", "png", "tif", "tiff"], key=f"f1_{run_id}")
        if f1:
            if st.session_state.processed_f1 != f1.name:
                st.session_state.images_analyzed += 1
                st.session_state.processed_f1 = f1.name
            st.session_state.compare_img1 = Image.open(io.BytesIO(f1.read()))
            
    with up_col2:
        f2 = st.file_uploader(t["upload_img_2"], type=["jpg", "jpeg", "png", "tif", "tiff"], key=f"f2_{run_id}")
        if f2:
            if st.session_state.processed_f2 != f2.name:
                st.session_state.images_analyzed += 1
                st.session_state.processed_f2 = f2.name
            st.session_state.compare_img2 = Image.open(io.BytesIO(f2.read()))

    if st.session_state.compare_img1 and st.session_state.compare_img2:
        w1, h1 = st.session_state.compare_img1.size
        w2, h2 = st.session_state.compare_img2.size
        if (w1 != w2) or (h1 != h2):
            st.session_state.compare_img2 = st.session_state.compare_img2.resize((w1, h1), Image.Resampling.LANCZOS)

        if st.session_state.compare_res1 is None or st.session_state.compare_res2 is None:
            with st.spinner(t["processing_spinner"]):
                res1, map1, grids1 = analyze_image_pixels(st.session_state.compare_img1)
                res2, map2, grids2 = analyze_image_pixels(st.session_state.compare_img2)

                st.session_state.compare_res1 = res1
                st.session_state.compare_res2 = res2
                st.session_state.compare_grids1 = grids1
                st.session_state.compare_grids2 = grids2
                st.session_state.compare_seg1 = generate_segmented_image(map1, st.session_state.compare_img1)
                st.session_state.compare_seg2 = generate_segmented_image(map2, st.session_state.compare_img2)
                st.session_state.last_processed_at = datetime.now()

    st.write("---")

    if st.session_state.compare_img1 or st.session_state.compare_img2:
        comp_view_mode = st.radio(t["select_view_mode"], (t["original_image"], t["segmented_view"]), horizontal=True, key="comp_view")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader(t["image_1_baseline"])
            if st.session_state.compare_img1:
                if hotspot_index != 0 and st.session_state.compare_grids1:
                    if hotspot_index == 1:
                        g_matrix = st.session_state.compare_grids1["other_grid"]
                    elif hotspot_index == 3:
                        g_matrix = st.session_state.compare_grids1["vegetation_grid"]
                    else: 
                        g_matrix = np.clip(100.0 - st.session_state.compare_grids1["vegetation_grid"], 0, 100)
                    
                    rendered_frame = generate_hotspot_overlay(st.session_state.compare_img1, g_matrix, hotspot_label_selected, overlay_intensity)
                    st.image(rendered_frame, use_container_width=True)
                elif comp_view_mode == t["original_image"]:
                    st.image(st.session_state.compare_img1, use_container_width=True)
                elif comp_view_mode == t["segmented_view"] and st.session_state.compare_seg1:
                    st.image(st.session_state.compare_seg1, use_container_width=True)

        with v_col2:
            st.subheader(t["image_2_comparison"])
            if st.session_state.compare_img2:
                if hotspot_index != 0 and st.session_state.compare_grids2:
                    if hotspot_index == 1:
                        g_matrix_2 = st.session_state.compare_grids2["other_grid"]
                    elif hotspot_index == 3:
                        g_matrix_2 = st.session_state.compare_grids2["vegetation_grid"]
                    else: 
                        g_matrix_2 = np.clip(100.0 - st.session_state.compare_grids2["vegetation_grid"], 0, 100)
                        
                    rendered_frame_2 = generate_hotspot_overlay(st.session_state.compare_img2, g_matrix_2, hotspot_label_selected, overlay_intensity)
                    st.image(rendered_frame_2, use_container_width=True)
                elif comp_view_mode == t["original_image"]:
                    st.image(st.session_state.compare_img2, use_container_width=True)
                elif comp_view_mode == t["segmented_view"] and st.session_state.compare_seg2:
                    st.image(st.session_state.compare_seg2, use_container_width=True)

        if hotspot_index != 0:
            render_weather_legend_html(hotspot_index, t)

    if st.session_state.compare_res1 and st.session_state.compare_res2:
        st.write("---")
        
        r1 = st.session_state.compare_res1
        r2 = st.session_state.compare_res2

        veg_delta = r2["vegetation_pct"] - r1["vegetation_pct"]
        wat_delta = r2["water_pct"] - r1["water_pct"]
        oth_delta = r2["other_pct"] - r1["other_pct"]
        soil_delta = r2["soil_pct"] - r1["soil_pct"]
        sand_delta = r2["sand_pct"] - r1["sand_pct"]

        veg_status_color = '#b91c1c' if veg_delta < 0 else '#166534'
        veg_status_text = f"{t['biomass_loss']} ({veg_delta:+.2f}%)" if veg_delta < 0 else f"{t['biomass_growth']} ({veg_delta:+.2f}%)"
        
        wat_status_color = '#b91c1c' if r2['water_pct'] <= 0 else ('#b91c1c' if wat_delta < 0 else '#166534')
        wat_status_text = t['critical_dryout'] if r2['water_pct'] <= 0 else (f"{t['hydrological_loss']} ({wat_delta:+.2f}%)" if wat_delta < 0 else t['stable_volume'])
        
        # Split logic into clean separate statements to avoid Python ternary parser issues
        Math_Check = (oth_delta <= 0)
        sprawl_status_color = '#166534' if Math_Check else '#b91c1c'
        sprawl_status_text = t['controlled_dev'] if Math_Check else t['expansion_spike']

        soil_status_color = '#b91c1c' if soil_delta > 0 else '#166534'
        soil_status_text = t['topsoil_exposure'] if soil_delta > 0 else t['stabilized_surface']

        classes = [
            ("vegetation_pct", t["vegetation"], "#2ecc71"),
            ("water_pct", t["water"], "#2980b9"),
            ("other_pct", t["other"], "#95a5a6"),
            ("soil_pct", t["soil"], "#8b4513"),
            ("sand_pct", t["sand"], "#d7b996")
        ]

        table_rows_html = ""
        for key, display_label, bar_color in classes:
            v1 = r1[key]
            v2 = r2[key]
            delta = v2 - v1

            trend_marker = "▲" if delta > 0 else "▼"
            
            if key == "water_pct" and v2 == 0.0:
                badge_style = "border: 1px dashed #ef4444; background: rgba(239, 68, 68, 0.08); color: #b91c1c;"
            elif (key in ["vegetation_pct", "water_pct"] and delta >= 0) or (key in ["other_pct", "soil_pct", "sand_pct"] and delta <= 0):
                badge_style = "background: rgba(34, 197, 94, 0.1); color: #166534;"
            else:
                badge_style = "background: rgba(254, 242, 242, 1); color: #b91c1c; border: 1px solid rgba(254, 226, 226, 1);"

            table_rows_html += f"""
            <tr>
                <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; font-weight: 700; color: #0f172a; font-size: 1.05rem;">{display_label}</td>
                <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #1e293b; font-size: 1.1rem; vertical-align: middle;">
                    <div style="font-weight: 600; margin-bottom: 4px;">{v1:.2f}%</div>
                    <div style="background: #e2e8f0; border-radius: 2px; height: 4px; width: 80px;">
                        <div style="background: {bar_color}; height: 100%; width: {min(v1, 100.0)}%; border-radius: 2px;"></div>
                    </div>
                </td>
                <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #1e293b; font-size: 1.1rem; vertical-align: middle;">
                    <div style="font-weight: 600; margin-bottom: 4px;">{v2:.2f}%</div>
                    <div style="background: #e2e8f0; border-radius: 2px; height: 4px; width: 80px;">
                        <div style="background: {bar_color}; height: 100%; width: {min(v2, 100.0)}%; border-radius: 2px;"></div>
                    </div>
                </td>
                <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; vertical-align: middle;">
                    <span style="padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.95rem; display: inline-block; {badge_style}">
                        {trend_marker} {delta:+.2f}%
                    </span>
                </td>
            </tr>
            """

        integrated_dashboard_html = f"""
        <div style="font-family: 'Segoe UI', system-ui, sans-serif; background: transparent; padding: 2px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;">
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 6px;">{t['card_veg']}</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">{veg_delta:+.2f}%</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {veg_status_color};">{veg_status_text}</div>
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 6px;">{t['card_water']}</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">{wat_delta:+.2f}%</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {wat_status_color};">{wat_status_text}</div>
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 6px;">{t['card_sprawl']}</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">{oth_delta:+.2f}%</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {sprawl_status_color};">{sprawl_status_text}</div>
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; font-weight: 800; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 6px;">{t['card_soil']}</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">{soil_delta:+.2f}%</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {soil_status_color};">{soil_status_text}</div>
                </div>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.01); overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; text-align: left; margin: 0;">
                    <thead>
                        <tr style="background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                            <th style="padding: 12px 16px; font-weight: 700; text-transform: uppercase; font-size: 0.8rem; color: #475569; letter-spacing: 0.05em;">{t['metric_label']}</th>
                            <th style="padding: 12px 16px; font-weight: 700; text-transform: uppercase; font-size: 0.8rem; color: #475569; letter-spacing: 0.05em;">{t['img1_label']}</th>
                            <th style="padding: 12px 16px; font-weight: 700; text-transform: uppercase; font-size: 0.8rem; color: #475569; letter-spacing: 0.05em;">{t['img2_label']}</th>
                            <th style="padding: 12px 16px; font-weight: 700; text-transform: uppercase; font-size: 0.8rem; color: #475569; letter-spacing: 0.05em;">{t['diff_label']}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """
        components.html(integrated_dashboard_html, height=540, scrolling=False)
        
        with st.container():
            render_dynamic_mitigation_suggestions(r2, t)

def main() -> None:
    st.set_page_config(page_title="Terra Drone", page_icon="🛰️", layout="wide")
    init_session_state()

    if st.session_state.selected_language is None:
        render_language_selection()
    elif st.session_state.selected_mode is None:
        render_mode_selection(TRANSLATIONS[st.session_state.selected_language])
    else:
        t = TRANSLATIONS[st.session_state.selected_language]
        render_sidebar(t)
        if st.session_state.selected_mode == "single":
            render_main(t)
        elif st.session_state.selected_mode == "compare":
            render_comparison(t)

if __name__ == "__main__":
    main()
